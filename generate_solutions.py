from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_tasks import _safe_write_text
from generator.apptainer_build import (
    format_apptainer_build_error,
    run_apptainer_build,
)
from generator.sample_solutions import run_n_solutions


@dataclass
class SolutionConfig:
    """Configuration for running solutions on tasks."""

    tasks_dir: str
    num_solutions: int = 128
    max_actions: int = 16
    model: str = "Qwen/Qwen3-32B"
    solution_temperature: float = 1.0
    verbose: bool = False
    num_tasks: int = 1
    start_at: int = 0
    num_pool_workers: int = 128
    workers: int = 1
    force_build: bool = False
    vllm: bool = False
    max_tokens: int = 2048
    filter_solved: bool = False
    use_parquet: bool = False


def _resolve_base_sif(def_path: Path) -> Optional[Path]:
    """Find the ubuntu_22.04.sif base image relative to the def or project root."""
    candidates = [
        def_path.parent / "ubuntu_22.04.sif",
        Path(__file__).resolve().parent / "ubuntu_22.04.sif",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _patch_def_text(def_text: str, def_path: Path) -> str:
    """Patch a def file's Bootstrap/From lines and ensure /home/user permissions."""
    import re

    base_sif = _resolve_base_sif(def_path)
    if base_sif:
        def_text = re.sub(
            r"^Bootstrap:.*$",
            "Bootstrap: localimage",
            def_text,
            count=1,
            flags=re.MULTILINE,
        )
        def_text = re.sub(
            r"^From:.*$",
            f"From: {base_sif}",
            def_text,
            count=1,
            flags=re.MULTILINE,
        )

    if "chmod 755 /home/user" not in def_text:
        import re as _re
        # Insert at the end of %post, right before the next section header
        def_text = _re.sub(
            r"(%post\b.*?)((?=\n%[a-z])|\Z)",
            r"\1\n    chmod 755 /home/user\n",
            def_text,
            count=1,
            flags=_re.DOTALL,
        )

    return def_text


def build_and_test(
    sif_path: Path, def_path: Path, test_py: str, run_initial_tests: bool = True
) -> tuple[bool, str]:
    """Build container and optionally run initial tests."""
    import tempfile

    with open(def_path, "r") as f:
        def_text = f.read()

    def_text = _patch_def_text(def_text, def_path)

    patched = Path(tempfile.mktemp(suffix=".def"))
    patched.write_text(def_text)

    try:
        build_proc = run_apptainer_build(sif_path, patched, timeout=180)
        if build_proc.returncode:
            err = format_apptainer_build_error(
                sif_path=sif_path,
                def_path=patched,
                returncode=build_proc.returncode,
                stdout=build_proc.stdout,
                stderr=build_proc.stderr,
            )
            print(err)
            return False, err
    except FileNotFoundError as exc:
        err = format_apptainer_build_error(
            sif_path=sif_path,
            def_path=patched,
            error=exc,
        )
        print(err)
        return False, err
    except subprocess.TimeoutExpired as exc:
        err = format_apptainer_build_error(
            sif_path=sif_path,
            def_path=patched,
            error=exc,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )
        print(err)
        return False, err
    finally:
        patched.unlink(missing_ok=True)

    if not run_initial_tests:
        return True, ""

    test_file = sif_path.parent / "test_initial_state.py"
    test_file.write_text(test_py)

    proc = subprocess.run(
        [
            "apptainer", "exec",
            "--fakeroot",
            "--userns",
            "--writable-tmpfs",
            "--cleanenv",
            str(sif_path),
            "pytest", "-q",
            str(test_file.name),
        ],
        capture_output=True,
        text=True,
    )

    return proc.returncode == 0, proc.stdout + proc.stderr


def process_task(task_dir: str, cfg: SolutionConfig):
    """Process a single task: build, test, run solutions, and cleanup."""
    task_dir = Path(task_dir)
    print(f"\nProcessing task: {task_dir.name}")

    sif_path = task_dir / "container.sif"
    def_path = task_dir / "container.def"
    initial_test_path = task_dir / "test_initial_state.py"
    final_test_path = task_dir / "test_final_state.py"
    task_json_path = task_dir / "task.json"
    solutions_dir = task_dir / "solutions"

    print(f"{task_dir} sif_path: {sif_path}")
    pass_at_k = None

    try:
        print(f"[{task_dir.name}] Running {cfg.num_solutions} solutions...")
        solutions_dir.mkdir(exist_ok=True)

        summary = run_n_solutions(
            num_solutions=cfg.num_solutions,
            container_sif_path=str(sif_path),
            initial_test_path=str(initial_test_path),
            final_test_path=str(final_test_path),
            def_path=str(def_path),
            task_path=str(task_json_path),
            max_actions=cfg.max_actions,
            model=cfg.model,
            temperature=cfg.solution_temperature,
            max_tokens=cfg.max_tokens,
            save_dir=str(solutions_dir),
            verbose=cfg.verbose,
            num_pool_workers=cfg.num_pool_workers,
            run_initial_tests=False,
        )

        model_name = cfg.model.replace("/", "_")
        _safe_write_text(
            task_dir / "solutions" / f"{model_name}_summary.json",
            json.dumps(summary, indent=4),
        )
        pass_at_k = summary.get("pass_at_k", {})

    finally:
        if sif_path.exists():
            print(f"[{task_dir.name}] Not deleting SIF file.")

    return pass_at_k


def parse_args(argv: Optional[List[str]] = None) -> SolutionConfig:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Build, test, and run solutions for generated tasks."
    )
    ap.add_argument(
        "--tasks-dir",
        type=str,
        default="/scr/kanishkg/os-tasks-o3-0901",
        help="Directory containing generated tasks",
    )
    ap.add_argument("--start-at", type=int, default=0, help="Start at task number")
    ap.add_argument("--num-tasks", type=int, default=200, help="Number of tasks to process")
    ap.add_argument(
        "--num-solutions", type=int, default=16, help="Number of solution attempts per task"
    )
    ap.add_argument(
        "--max-actions", type=int, default=16, help="Max shell actions per solution attempt"
    )
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    ap.add_argument("--solution-temperature", type=float, default=1.0)
    ap.add_argument(
        "--max-tokens", type=int, default=2048, help="Max tokens for the solution agent"
    )
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    ap.add_argument("--num-pool-workers", type=int, default=128, help="Number of pool workers")
    ap.add_argument(
        "--workers", type=int, default=1, help="Number of concurrent tasks to process"
    )
    ap.add_argument("--force-build", action="store_true", help="Force build the SIF file")
    ap.add_argument(
        "--filter-solved", action="store_true", help="Only solve tasks that have been solved by o3"
    )
    ap.add_argument("--use-parquet", action="store_true", help="Use parquet file for tasks")

    args = ap.parse_args(argv)
    return SolutionConfig(**vars(args))


def main():
    """Main entry point for solution generation."""
    cfg = parse_args()

    all_entries = list(Path(cfg.tasks_dir).iterdir())
    task_dirs = [
            d
            for d in tqdm(all_entries, desc="Scanning task directories", total=len(all_entries))
            if d.name.startswith("task_")
        ]
    if cfg.filter_solved:

        print(f"Filtering to tasks with o3 pass@16 > 0, prefilter: {len(task_dirs)}")

        def _o3_pass16_gt_zero(task_dir: str) -> bool:
            task_dir = Path(task_dir)
            try:
                summary_path = task_dir / "solutions" / "o3_summary.json"
                if not summary_path.exists():
                    return False
                model_name = cfg.model.replace("/", "_")
                model_summary_path = task_dir / "solutions" / f"{model_name}_summary.json"
                if model_summary_path.exists():
                    return False
                with open(summary_path, "r") as f:
                    data = json.load(f)
                pass_at_k = data.get("pass_at_k", {})
                # Keys may be strings or ints
                value = pass_at_k.get("16") or pass_at_k.get(16)
                if value is None:
                    return False
                return float(value) > 0.0
            except Exception:
                return False

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=len(task_dirs)) as executor:
            futures = {executor.submit(_o3_pass16_gt_zero, d): i for i, d in enumerate(task_dirs)}
            mask = [False] * len(task_dirs)
            with tqdm(total=len(task_dirs), desc="Reading o3 summaries") as pbar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        mask[idx] = fut.result()
                    except Exception:
                        mask[idx] = False
                    finally:
                        pbar.update(1)
        task_dirs = [d for d, ok in zip(task_dirs, mask) if ok]

        print(f"Filtering to tasks with o3 pass@16 > 0, postfilter: {len(task_dirs)}")
        time.sleep(30)

    if cfg.use_parquet:
        from datasets import load_dataset

        dataset = load_dataset(
            "parquet", data_files=os.path.join(cfg.tasks_dir, "train.parquet")
        )["train"]
        task_dirs = [d["extra_info"]["task_dir"] for d in dataset]

    task_dirs = list(sorted(task_dirs))
    task_dirs = task_dirs[cfg.start_at : min(cfg.start_at + cfg.num_tasks, len(task_dirs))]

    if not task_dirs:
        print(f"No task directories found in {cfg.tasks_dir}")
        return

    def process_task_with_retry(task_dir: str, cfg: SolutionConfig):
        """Wrap per-task retry logic so it can run in parallel."""
        task_dir = Path(task_dir)
        max_retries = 2
        result = None

        for attempt in range(max_retries):
            try:
                result = process_task(task_dir, cfg)
            except Exception as e:
                print(f"[{task_dir.name}] Attempt {attempt + 1}/{max_retries} failed with exception: {e}")
                result = None

            if result is None:
                if attempt < max_retries - 1:
                    print(f"[{task_dir.name}] Retrying...")
                else:
                    print(f"[{task_dir.name}] All {max_retries} attempts failed, skipping.")
            elif result in ("no def", "no sif", "no initial test"):
                print(f"[{task_dir.name}] Missing files ({result}), skipping.")
                break
            else:
                print(f"[{task_dir.name}] Pass@k: {result}")
                break

        return task_dir, result

    # Cap workers so total concurrent containers stays sane.
    # Each worker runs num_solutions containers simultaneously.
    max_total_instances = 32
    effective_workers = max(1, min(cfg.workers, max_total_instances // max(1, cfg.num_solutions)))
    if effective_workers != cfg.workers:
        print(
            f"Capping --workers from {cfg.workers} to {effective_workers} "
            f"({effective_workers} × {cfg.num_solutions} solutions = "
            f"{effective_workers * cfg.num_solutions} concurrent containers)"
        )

    if effective_workers <= 1:
        for task_dir in tqdm(task_dirs, desc="Processing Tasks"):
            process_task_with_retry(task_dir, cfg)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(process_task_with_retry, td, cfg): td for td in task_dirs}
            with tqdm(total=len(task_dirs), desc="Processing Tasks") as pbar:
                for fut in as_completed(futures):
                    try:
                        _td, _res = fut.result()
                    finally:
                        pbar.update(1)


if __name__ == "__main__":
    main()
