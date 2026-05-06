from __future__ import annotations

import getpass
import os
import shlex
import socket
import subprocess
from pathlib import Path
from typing import Optional

SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SANITIZE_ENV_KEYS = (
    "LD_PRELOAD",
    "APPTAINERENV_LD_PRELOAD",
    "SINGULARITYENV_LD_PRELOAD",
    "FAKEROOTKEY",
)


def sanitized_apptainer_env() -> dict[str, str]:
    """Return a host environment safe for invoking Apptainer builds."""
    env = os.environ.copy()
    for key in SANITIZE_ENV_KEYS:
        env.pop(key, None)
    env["PATH"] = env.get("PATH") or SAFE_PATH
    return env


def run_apptainer_build(
    sif_path: Path,
    def_path: Path,
    *,
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    """Run `apptainer build` with a sanitized host environment."""
    cmd = ["apptainer", "build", str(sif_path), str(def_path)]
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=sanitized_apptainer_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def format_apptainer_build_error(
    *,
    sif_path: Path,
    def_path: Path,
    returncode: Optional[int] = None,
    stdout: str = "",
    stderr: str = "",
    error: Optional[BaseException] = None,
    cwd: Optional[Path] = None,
) -> str:
    """Format a build failure with enough host context to debug fakeroot issues."""
    user = getpass.getuser()
    uid = os.getuid()
    path_value = os.environ.get("PATH") or SAFE_PATH
    cleared_env = [key for key in SANITIZE_ENV_KEYS if key in os.environ]
    cmd = ["apptainer", "build", str(sif_path), str(def_path)]

    lines = [
        "Apptainer build failed.",
        f"cmd: {shlex.join(cmd)}",
        f"host: {socket.gethostname()}",
        f"user: {user} (uid={uid})",
        f"cwd: {cwd or Path.cwd()}",
        f"PATH: {_clip(path_value, 240)}",
        f"/etc/subuid: {_read_subid_entry(Path('/etc/subuid'), user, uid)}",
        f"/etc/subgid: {_read_subid_entry(Path('/etc/subgid'), user, uid)}",
        f"cleared_env: {', '.join(cleared_env) if cleared_env else 'none'}",
    ]
    if returncode is not None:
        lines.append(f"returncode: {returncode}")
    if error is not None:
        lines.append(f"error: {error}")

    combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part).strip()
    if combined:
        lines.append("output tail:")
        lines.append(_clip(combined, 2000))

    return "\n".join(lines)


def _read_subid_entry(path: Path, username: str, uid: int) -> str:
    try:
        if not path.exists():
            return "missing"
        matches = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{username}:") or line.startswith(f"{uid}:"):
                matches.append(line)
        return " | ".join(matches) if matches else "no entry"
    except OSError as exc:
        return f"unreadable ({exc})"


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]
