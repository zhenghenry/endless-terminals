# generator/completion_test_gen.py
"""Generate a pytest *template* that validates the **final** state after the task
is completed.

This script consumes the task-template JSON produced by ``task_template_gen.py``
(which contains the description template, parameter schema and a privileged
``truth`` section).  It samples concrete values for each placeholder, renders a
full *task description* and then asks the LLM to create a single pytest file
(`test_final_state.py`) that passes **only if** the task has been solved
correctly.  The privileged ``truth`` data is forwarded to the LLM so the tests
can assert the exact expected end state.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
import sys
from typing import Optional

# Ensure the project root is on ``sys.path`` so ``generator.chat_completion`` can
# be imported when this script is executed from arbitrary working directories.
sys.path.insert(0, str(Path().resolve()))
from generator import  parse_python_code, check_python_code, chat_completion_batch

# ---------------------------------------------------------------------------
# LLM prompt scaffolding
# ---------------------------------------------------------------------------

SYSTEM_MSG = """You are a senior Python engineer who writes robust pytest suites.
Write a robust pytest suite that validates the **FINAL** state of the operating-system / container **after** the student has
completed the task described.
Use the privileged *truth* data to assert the exact expected end state for the task to be completed.

Rules:
* The filename must be ``test_final_state.py`` (show it in a header comment).
* Use **only** the Python standard library and ``pytest`` (no third-party libs).
* Failures must clearly explain **what is still wrong**.
* When you check for files or directories, always use their *absolute* paths exactly as given (no relative paths).
* Ensure that the the state of the OS matches the truth after the task is completed.
* Write the code in a fenced code block that can be parsed to get a single python file."""

USER_TEMPLATE = """The task description is: {task_description}
The truth value is: {truth}
The tests to check the initial container state, before the task is completed, are:
{initial_test_py}
Write the code in a fenced code block that can be parsed."""

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def generate_test_templates_batch(
    items: list[tuple[str, str, str]],
    *,
    model: str = "qwen/Qwen2.5-3B-Instruct",
    temperature: float = 0.6,
    max_tokens: int = 2048,
    max_concurrency: int = 128,
) -> list[Optional[str]]:
    """Batched generation of final-state pytest templates.

    items: list of (task_description, truth, initial_test_py). Returns aligned list with None on failure.
    """

    messages: list[list[dict[str, str]]] = []
    for task_description, truth, initial_test_py in items:
        prompt = USER_TEMPLATE.format(task_description=task_description, truth=truth, initial_test_py=initial_test_py)
        messages.append([
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ])

    responses = chat_completion_batch(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_completions=1,
        max_concurrency=max_concurrency,
    )

    results: list[Optional[str]] = []
    for resp in responses:
        if resp is None:
            print("Response is None")
            results.append(None)
            continue
        try:
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                print("Test template truncated (hit max_tokens limit)")
                results.append(None)
                continue
            content = textwrap.dedent(choice.message.content)
            parsed = parse_python_code(content)
            if check_python_code(parsed):
                results.append(parsed)
            else:
                print("Test template is not valid")
                results.append(None)
        except Exception as e:
            print(f"Error parsing test template: {e}")
            results.append(None)
    return results


