"""Environment interaction module for agent testing in Apptainer containers with interactive shell (PTY-backed, promptless)."""
from __future__ import annotations

import errno
import fcntl
import json
import os
import pty
import queue
import re
import shutil
import subprocess
import tempfile
import termios
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from generator.apptainer_build import (
    format_apptainer_build_error,
    run_apptainer_build,
)

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")  # Strip ANSI escapes

_dbus_setup_lock = threading.Lock()
_dbus_pid: Optional[int] = None


def _ensure_dbus_session() -> None:
    """Start a dbus session daemon if one isn't available for the current user.

    Apptainer 1.4+ on cgroups v2 requires a dbus session bus to manage
    cgroups for rootless instances.  Non-login sessions (SSH, service
    accounts) often lack one.
    """
    global _dbus_pid
    with _dbus_setup_lock:
        uid = os.getuid()
        default_sock = f"/run/user/{uid}/bus"

        # Already available via the standard path
        if os.path.exists(default_sock):
            if "DBUS_SESSION_BUS_ADDRESS" not in os.environ:
                os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={default_sock}"
            return

        # Already set up by us in a previous call
        addr = os.environ.get("DBUS_SESSION_BUS_ADDRESS", "")
        if addr and os.path.exists(addr.replace("unix:path=", "")):
            return

        # Start a private dbus-daemon with a temp socket
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/tmp/apptainer-dbus-{uid}")
        os.makedirs(runtime_dir, exist_ok=True)
        sock_path = os.path.join(runtime_dir, "bus")

        try:
            proc = subprocess.run(
                ["dbus-daemon", "--session", f"--address=unix:path={sock_path}",
                 "--fork", "--print-pid"],
                capture_output=True, text=True,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                _dbus_pid = int(proc.stdout.strip())
                os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={sock_path}"
        except FileNotFoundError:
            pass


class InteractiveContainerEnvironment:
    """Manages interaction with a pre-built Apptainer container using an interactive shell over a PTY."""

    def __init__(
        self,
        container_sif_path: str,
        initial_test_path: str,
        final_test_path: str,
        def_path: str,
        max_actions: int = 50,
        verbose: bool = True,
        read_timeout: float = 10.0,
    ):
        # Resolve all incoming paths to absolute paths
        self.sif_path = Path(container_sif_path).expanduser().resolve()
        self.initial_test_path = Path(initial_test_path).expanduser().resolve()
        self.final_test_path = Path(final_test_path).expanduser().resolve()
        self.def_path = Path(def_path).expanduser().resolve()

        self.max_actions = max_actions
        self.verbose = verbose
        self.read_timeout = read_timeout

        self.temp_dir: Optional[Path] = None
        self.action_history: List[Dict[str, str]] = []
        self.instance_name: Optional[str] = None

        self.shell_process: Optional[subprocess.Popen] = None
        self.master_fd: Optional[int] = None
        self.slave_fd: Optional[int] = None

        self.output_queue: "queue.Queue[str]" = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Unique marker to delimit command completion and carry exit code
        self._marker = f"__CMD_DONE__{uuid.uuid4().hex}__"

    # ----------------------------
    # Low-level PTY I/O utilities
    # ----------------------------
    def _reader_loop(self) -> None:
        """Background thread to read from PTY master and push text into a queue (no selector)."""
        fd = self.master_fd
        if fd is None:
            return
        # Non-blocking
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while (
                not self._stop_event.is_set()
                and self.shell_process
                and self.shell_process.poll() is None
            ):
                try:
                    data = os.read(fd, 16384)
                    if data:
                        text = data.decode("utf-8", errors="replace")
                        self.output_queue.put_nowait(text)
                        continue
                except BlockingIOError:
                    pass
                except OSError as e:
                    if getattr(e, "errno", None) in (errno.EBADF, errno.EIO):
                        break
                    raise
                time.sleep(0.005)
        finally:
            try:
                while True:
                    data = os.read(fd, 16384)
                    if not data:
                        break
                    text = data.decode("utf-8", errors="replace")
                    self.output_queue.put_nowait(text)
            except Exception:
                pass

    def _drain_queue(self) -> str:
        chunks: List[str] = []
        while True:
            try:
                chunks.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return "".join(chunks)

    def _read_until_marker(self, timeout: Optional[float] = None) -> Tuple[str, Optional[int]]:
        """
        Read buffered output until we see our unique marker line, e.g. '__CMD_DONE__...__:0'
        Returns (output_without_marker, exit_code or None on timeout)
        """
        if timeout is None:
            timeout = self.read_timeout

        deadline = time.time() + timeout
        buf = []

        marker_match = None
        while time.time() < deadline:
            # pull whatever we have
            chunk = self._drain_queue()
            if chunk:
                buf.append(chunk)
                joined = "".join(buf)
                # try to find the last marker occurrence (in case command prints similar text)
                for line in joined.splitlines():
                    if self._marker in line:
                        # marker format: {marker}:{exit_code}
                        if ":" in line:
                            parts = line.rsplit(":", 1)
                            if len(parts) == 2 and parts[0].endswith(self._marker):
                                try:
                                    code = int(parts[1].strip())
                                except ValueError:
                                    code = None
                                marker_match = (joined, code)
                if marker_match:
                    full_out, code = marker_match
                    first_marker_index = full_out.find(self._marker)
                    cleaned = full_out[:first_marker_index]
                    return cleaned, code
            time.sleep(0.002)

        # timeout: return whatever we accumulated
        return "".join(buf), None

    # ----------------------------
    # Shell lifecycle
    # ----------------------------
    def _start_shell(self) -> bool:
        """Start an interactive Apptainer shell session on a PTY."""
        if self.shell_process:
            return True

        # Create PTY pair
        self.master_fd, self.slave_fd = pty.openpty()

        # Make the slave a proper TTY with sane settings
        try:
            attrs = termios.tcgetattr(self.slave_fd)
            # disable echo (we'll still see command output, avoids double-echo)
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(self.slave_fd, termios.TCSANOW, attrs)
        except Exception:
            pass

        # Compose apptainer shell command
        cmd = [
            "apptainer", "shell",
            "--containall",
            "--cleanenv",
            "--pwd", "/home/user",
            f"instance://{self.instance_name}",
        ]

        try:
            # Launch with PTY endpoints
            self.shell_process = subprocess.Popen(
                cmd,
                stdin=self.slave_fd,
                stdout=self.slave_fd,
                stderr=self.slave_fd,
                close_fds=True,
                start_new_session=True,  # new process group
            )

            # Start reader thread
            self._stop_event.clear()
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

            # Give shell a brief moment to start up
            time.sleep(0.05)
            if self.shell_process.poll() is not None:
                self._drain_queue()
                return False

            # Initialize shell: make it predictable
            init_script = (
                "set -o pipefail 2>/dev/null; "
                "export PS1='[$PWD]$ '; "
                "export HOME=/home/user; "
                "cd \"$HOME\" 2>/dev/null || true; "
                f"printf '{self._marker}:0\\n'"
            )
            os.write(self.master_fd, (init_script + "\n").encode("utf-8"))
            _, code = self._read_until_marker(timeout=10.0)
            if code is None:
                return False

            self._drain_queue()
            return True
        except Exception:
            return False

    def _stop_shell(self):
        """Stop the interactive shell session and close PTY."""
        try:
            # signal reader to stop before we close fds
            self._stop_event.set()
            if self.reader_thread:
                try:
                    self.reader_thread.join(timeout=1.0)
                except Exception:
                    pass
                self.reader_thread = None

            if self.shell_process and self.shell_process.poll() is None:
                try:
                    os.write(self.master_fd, b"exit\n")
                except Exception:
                    pass
                try:
                    self.shell_process.wait(timeout=2)
                except Exception:
                    self.shell_process.terminate()
                    try:
                        self.shell_process.wait(timeout=2)
                    except Exception:
                        self.shell_process.kill()
        finally:
            self.shell_process = None

            # Close PTY fds
            for fd in (self.master_fd, self.slave_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
            self.master_fd = None
            self.slave_fd = None

    def _stop_instance(self) -> None:
        """Stop the Apptainer instance if running."""
        if self.instance_name:
            subprocess.run(
                ["apptainer", "instance", "stop", self.instance_name],
                capture_output=True
            )
            self.instance_name = None

    # ----------------------------
    # Public API
    # ----------------------------
    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def initialize(self, run_initial_tests: bool = True) -> bool:
        """Initialize the container environment and validate initial state."""
        if not self.sif_path.exists():
            if self.def_path.exists():
                self.build_container()
            else:
                return False

        self.temp_dir = Path(tempfile.mkdtemp(prefix="agent_env_")).resolve()
        _ensure_dbus_session()

        self.instance_name = f"agent_{uuid.uuid4().hex[:8]}"
        start_cmd = [
            "apptainer", "instance", "start",
            "--containall",
            "--writable-tmpfs",
            "--bind", f"{self.temp_dir}:{self.temp_dir}",
            "--cleanenv",
            str(self.sif_path),
            self.instance_name,
        ]
        start_proc = subprocess.run(start_cmd, capture_output=True, text=True)
        if start_proc.returncode != 0:
            return False

        if not self._start_shell():
            self._stop_instance()
            return False

        if run_initial_tests:
            if not self.run_initial_tests():
                self._stop_shell()
                self._stop_instance()
                return False

        self.exec("cd /home/user")
        return True

    def exec(self, command: str, timeout: Optional[float] = None) -> Tuple[bool, str]:
        """
        Execute a command in the interactive shell.

        Returns:
            (success, output_without_marker_and_ansi)
        """
        # Check if shell exists
        if not self.shell_process:
            if not self._start_shell():
                return False, "Failed to start shell"

        # Check if shell process is still alive
        if self.shell_process.poll() is not None:
            self.shell_process = None
            if not self._start_shell():
                return False, "Shell process died and restart failed"

        if not self.reader_thread or not self.reader_thread.is_alive():
            self._stop_shell()
            if not self._start_shell():
                return False, "Reader thread died and shell restart failed"

        # Clear any stale output
        _ = self._drain_queue()

        # Wrap the command to always emit our marker with the exit code
        # Use a subshell to ensure we capture the correct `$?` across pipelines
        # Special-case heredocs: avoid grouping with braces so the terminator can be on its own line
        if "<<" in command:
            wrapped = f"{command}\ncode=$?; printf '{self._marker}:%s\\n' \"$code\""
        else:
            wrapped = f"{{ {command}; }}; code=$?; printf '{self._marker}:%s\\n' \"$code\""
        
        try:
            os.write(self.master_fd, (wrapped + "\n").encode("utf-8"))
        except Exception as e:
            return False, f"Command write failed: {e}"

        time.sleep(0.01)
        if self.shell_process.poll() is not None:
            return False, f"Shell died immediately after command (exit code: {self.shell_process.returncode})"

        raw_out, code = self._read_until_marker(timeout=timeout)
        
        # Handle timeout
        if code is None:
            return False, f"Command timed out. Partial output:\n{raw_out[:500]}"
        
        # Clean output: strip ANSI, strip echoed lines (PTY has no echo, but some programs add it)
        cleaned = ANSI_RE.sub("", raw_out)
        cleaned = cleaned.replace("\r", "")

        success = (code == 0)
        return success, cleaned

    def run_initial_tests(self) -> bool:
        """Run initial state validation tests."""
        with open(self.initial_test_path, "r") as f:
            test_file_text = f.read()

        test_path_in_container = "/home/user/test_initial.py"
        marker = f"EOF_TEST_FILE_{uuid.uuid4().hex}"
        write_cmd = (
            f"cat <<'{marker}' > {test_path_in_container}\n"
            f"{test_file_text}\n"
            f"{marker}\n"
        )

        success, output = self.exec(write_cmd)
        if not success:
            return False

        test_success, test_output = self.exec(f"pytest -q {test_path_in_container}")
        self.exec(f"rm -f {test_path_in_container}")
        return test_success

    def run_final_tests(self) -> Tuple[bool, str]:
        """Run final state validation tests inside the instance."""
        with open(self.final_test_path, "r") as f:
            test_file_text = f.read()

        test_path_in_container = "/home/user/test_final.py"
        marker = f"EOF_TEST_FILE_{uuid.uuid4().hex}"
        write_cmd = (
            f"cat <<'{marker}' > {test_path_in_container}\n"
            f"{test_file_text}\n"
            f"{marker}\n"
        )
        ok, write_out = self.exec(write_cmd)
        if not ok:
            return False, write_out

        test_success, test_output = self.exec(f"pytest -q {test_path_in_container}")
        self.exec(f"rm -f {test_path_in_container}")
        return test_success, test_output

    def cleanup(self):
        """Clean up temporary files and processes."""
        self._stop_shell()
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self._stop_instance()

    def build_container(self):
        """Rebuild sif from def."""
        with open(self.def_path, "r") as f:
            def_text = f.read()

        # Resolve the base SIF to an absolute path so the build works
        # regardless of CWD.  Search next to the def file, in the project
        # root, and in common locations.
        base_sif = None
        candidates = [
            self.def_path.parent / "ubuntu_22.04.sif",
            Path(__file__).resolve().parent.parent / "ubuntu_22.04.sif",
        ]
        for c in candidates:
            if c.exists():
                base_sif = str(c.resolve())
                break

        if base_sif:
            import re as _re
            def_text = _re.sub(
                r"^Bootstrap:.*$",
                "Bootstrap: localimage",
                def_text,
                count=1,
                flags=_re.MULTILINE,
            )
            def_text = _re.sub(
                r"^From:.*$",
                f"From: {base_sif}",
                def_text,
                count=1,
                flags=_re.MULTILINE,
            )

        if "chmod 755 /home/user" not in def_text:
            import re as _re2
            def_text = _re2.sub(
                r"(%post\b.*?)((?=\n%[a-z])|\Z)",
                r"\1\n    chmod 755 /home/user\n",
                def_text,
                count=1,
                flags=_re2.DOTALL,
            )

        # Write a patched copy to a temp file so we don't corrupt the
        # original def (which may be read-only or shared).
        import tempfile as _tmpmod
        patched = Path(_tmpmod.mktemp(suffix=".def"))
        patched.write_text(def_text)

        try:
            build_rc = run_apptainer_build(self.sif_path, patched)
            if build_rc.returncode != 0:
                print(
                    format_apptainer_build_error(
                        sif_path=self.sif_path,
                        def_path=patched,
                        returncode=build_rc.returncode,
                        stdout=build_rc.stdout,
                        stderr=build_rc.stderr,
                    )
                )
                return False
            return True
        except FileNotFoundError as exc:
            print(
                format_apptainer_build_error(
                    sif_path=self.sif_path,
                    def_path=patched,
                    error=exc,
                )
            )
            return False
        except subprocess.TimeoutExpired as exc:
            print(
                format_apptainer_build_error(
                    sif_path=self.sif_path,
                    def_path=patched,
                    error=exc,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                )
            )
            return False
        finally:
            patched.unlink(missing_ok=True)

    def get_prompt(self) -> str:
        """Get the current shell prompt showing the working directory."""
        success, output = self.exec("pwd")
        if success and output:
            current_dir = output.strip().splitlines()[-1]
            return f"({self.sif_path.name}) {current_dir} $ "
        return f"({self.sif_path.name}) $ "


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=str, default="tasks/sample_task")
    args = parser.parse_args()
    task_path = Path(args.task_path)
    def_path = task_path / "container.def"
    initial_test_path = task_path / "test_initial_state.py"
    final_test_path = task_path / "test_final_state.py"
    container_sif_path = task_path / "container.sif"

    # sample_task.json is optional; guard it to avoid crashes
    task_description = ""
    truth = ""
    if Path("sample_task.json").exists():
        with open("sample_task.json", "r") as f:
            task_data = json.load(f)
        task_description = task_data.get("description", "")
        truth = task_data.get("truth", "")

    env = InteractiveContainerEnvironment(
        container_sif_path=container_sif_path,
        initial_test_path=initial_test_path,
        final_test_path=final_test_path,
        def_path=def_path,
        verbose=True,
    )
    if not container_sif_path.exists():
        env.build_container()

    if not env.initialize(run_initial_tests=True):
        raise SystemExit(1)

    try:
        print("\nStarting interactive session with the container...")
        print("Type 'exit' or 'quit' to finish.")
        if task_description:
            print(f"Task description: {task_description}")

        while True:
            try:
                prompt = env.get_prompt()
                command = input(prompt)
                if command.lower() in ["exit", "quit"]:
                    break
                if not command.strip():
                    continue
                success, output = env.exec(command)
                if output:
                    print(output)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive session.")
                break
    finally:
        env.run_final_tests()
        env.cleanup()
