"""Assert-line ablation tasks: how much of EvaLoop's ASL rests on the leaked example?

The standard loop re-attaches the original prompt's final assert line to every
summary (see :class:`evaloop.tasks.code_summarization.CodeSummarizationTask`).
That line exists to keep the function name and signature stable across cycles,
but it also re-injects a *worked example* — one concrete input with its expected
output — into every single cycle, from outside the loop. The model therefore
never has to recover the task's interface from its own summary, and it always
sees one ground-truth test case.

This module replaces that carrier with weaker ones, holding everything else
fixed, so the contribution of each piece of leaked information can be measured
separately:

==============  =====================================  ======================
arm             carrier appended to the summary        information retained
==============  =====================================  ======================
``assert``      ``assert f(3, 4) == 7``                name, arity, example
``signature``   ``def f(arg1, arg2):``                 name, arity
``name``        ``The function must be named `f`.``    name
``bare``        *(nothing)*                            nothing
==============  =====================================  ======================

Reading down that table is a monotone removal of information, so the ASL
differences between adjacent arms attribute the standard loop's score to the
example, the arity, and the name respectively.

Design notes
------------

*The carrier is always derived from the original prompt, never from a later
cycle.* ``signature_from_assert`` is applied once, to the assert line of the
cycle-1 input, and the result is memoised per ``task_id``. Later cycles reuse
that memo rather than re-parsing their own docstring. This keeps the arms
strictly comparable (every arm's carrier traces to the same source as the
baseline's) and prevents the carrier itself from drifting, which would conflate
interface drift with the semantic drift under study.

*Signature stability is solved, not dodged.* The ``signature`` and ``name`` arms
still pin the function name, so a failure means the model could not re-derive
the task from its own description — not that it renamed the function. The
``bare`` arm deliberately gives that up and is reported as the floor: it
measures the standard loop's total dependence on out-of-loop information.

``signature_from_assert`` was validated against all 378 MBPP Plus assert lines:
it recovers the entry point for 378/378. The three tasks whose cycle-1 code does
not define the recovered name are tasks the model failed at cycle 1.
"""

import ast
import logging
import re
import threading
from typing import Any, Dict, Optional, Tuple

from .base import BaseTask
from .factory import register_task

logger = logging.getLogger(__name__)

#: Builtins and test helpers that wrap the task's own call in an MBPP assert,
#: e.g. ``assert set(similar_elements(...)) == set(...)``. The task function is
#: the innermost non-wrapper call, so these are skipped when choosing it.
WRAPPERS = frozenset({
    "set", "tuple", "list", "dict", "frozenset", "sorted", "reversed", "len",
    "abs", "round", "str", "int", "float", "bool", "complex", "sum", "min",
    "max", "any", "all", "type", "repr", "next", "iter", "enumerate", "zip",
    "map", "filter", "range", "print", "divmod", "pow", "isclose", "approx",
})


def signature_from_assert(assert_line: str) -> Optional[Tuple[str, int]]:
    """Recover ``(function_name, n_args)`` from an MBPP Plus assert line.

    Args:
        assert_line: The ``assert ...`` line from a task prompt.

    Returns:
        The task function's name and argument count, or ``None`` if the line
        cannot be parsed.
    """
    src = assert_line.strip()
    if not src:
        return None
    if not src.startswith("assert"):
        src = "assert " + src

    tree = None
    for candidate in (src, src[: src.rfind(")") + 1] if ")" in src else ""):
        if not candidate:
            continue
        try:
            tree = ast.parse(candidate)
            break
        except SyntaxError:
            continue
    if tree is None:
        return None

    def collect(skip_wrappers: bool):
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if skip_wrappers and node.func.id in WRAPPERS:
                    continue
                found.append((node.col_offset, node.func.id,
                              len(node.args) + len(node.keywords)))
        return found

    # Prefer a non-builtin call. A few MBPP tasks name their function after a
    # builtin (Mbpp/126 is literally `sum`), so fall back to allowing wrappers
    # rather than failing to produce a carrier at all.
    candidates = collect(skip_wrappers=True) or collect(skip_wrappers=False)
    if not candidates:
        return None
    candidates.sort()                      # leftmost call wins
    _, name, n_args = candidates[0]
    return name, n_args


def _carrier_assert(assert_line: str, sig: Optional[Tuple[str, int]]) -> str:
    """Baseline carrier: the original assert line, verbatim."""
    return assert_line


def _carrier_signature(assert_line: str, sig: Optional[Tuple[str, int]]) -> str:
    """Name and arity, with the example values removed."""
    if sig is None:
        return ""
    name, n_args = sig
    args = ", ".join(f"arg{i + 1}" for i in range(n_args))
    return f"def {name}({args}):"


def _carrier_name(assert_line: str, sig: Optional[Tuple[str, int]]) -> str:
    """The function name only; arity must be recovered from the summary."""
    if sig is None:
        return ""
    return f"The function must be named `{sig[0]}`."


def _carrier_bare(assert_line: str, sig: Optional[Tuple[str, int]]) -> str:
    """Nothing: the floor arm."""
    return ""


CARRIERS = {
    "assert": _carrier_assert,
    "signature": _carrier_signature,
    "name": _carrier_name,
    "bare": _carrier_bare,
}


class AssertAblationSummarizationTask(BaseTask):
    """Summarisation whose docstring carrier is configurable.

    ``task_config["carrier"]`` selects one of :data:`CARRIERS`. The prompt
    template is identical across arms so that the carrier is the only
    manipulated variable.
    """

    #: Summarisation prompt from the course data package
    #: (``prompts/task_prompts.md``), used for every arm so that a locally run
    #: ``assert`` arm reproduces the published protocol rather than the
    #: repository's one-sentence default.
    DEFAULT_PROMPT = (
        "Summarize what the following code does in a detailed description:\n\n"
        "```\n{code}\n```\n\n"
        "Return only the summary, without explanations or comments."
    )

    def __init__(self, task_config: Dict[str, Any]):
        super().__init__(task_config)
        self.carrier_name = task_config.get("carrier", "assert")
        if self.carrier_name not in CARRIERS:
            raise ValueError(
                f"Unknown carrier '{self.carrier_name}'. "
                f"Available: {', '.join(sorted(CARRIERS))}"
            )
        self.carrier = CARRIERS[self.carrier_name]
        # task_id -> (name, n_args), parsed once from the cycle-1 assert line.
        # The loop runner shares one task instance across every prompt in a
        # cycle and across cycles, and batch mode may touch it from several
        # threads, so the memo is guarded.
        self._signatures: Dict[str, Optional[Tuple[str, int]]] = {}
        self._asserts: Dict[str, str] = {}
        self._lock = threading.Lock()

    def create_prompt(self, input_text: str) -> str:
        template = self.task_config.get("prompt_template", self.DEFAULT_PROMPT)
        return template.format(code=input_text)

    def extract_output(self, raw_response: str) -> str:
        return raw_response.strip()

    def _remember(self, task_id: str, cycle_input: str) -> Tuple[str, Optional[Tuple[str, int]]]:
        """Return the original assert line and signature for a task.

        Parsed from ``cycle_input`` the first time the task is seen — which is
        cycle 1, whose input is the untouched MBPP Plus prompt — and memoised.
        """
        with self._lock:
            if task_id in self._asserts:
                return self._asserts[task_id], self._signatures[task_id]

        lines = cycle_input.splitlines()
        # MBPP Plus prompt layout: `"""`, description, assert, `"""`.
        assert_line = lines[-2] if len(lines) >= 2 else ""
        sig = signature_from_assert(assert_line)
        if sig is None and assert_line:
            logger.warning(
                "Could not recover a signature for %s from %r; "
                "arm '%s' will carry nothing for this task",
                task_id, assert_line[:80], self.carrier_name,
            )

        with self._lock:
            self._asserts.setdefault(task_id, assert_line)
            self._signatures.setdefault(task_id, sig)
            return self._asserts[task_id], self._signatures[task_id]

    def postprocess(self, output: str, context: Dict[str, Any]) -> str:
        """Wrap the summary as a docstring plus this arm's carrier line."""
        task_id = context.get("task_id", "")
        cycle_input = context.get("cycle_input", "")
        assert_line, sig = self._remember(task_id, cycle_input)

        carrier = self.carrier(assert_line, sig)
        body = f"{output}\n{carrier}" if carrier else output
        return f'"""\n{body}\n"""\n'


@register_task("code_summarization_assert")
class AssertCarrierSummarizationTask(AssertAblationSummarizationTask):
    """Arm 1 — carries the original assert line (the standard loop)."""

    def __init__(self, task_config: Dict[str, Any]):
        super().__init__({**task_config, "carrier": "assert"})


@register_task("code_summarization_signature")
class SignatureCarrierSummarizationTask(AssertAblationSummarizationTask):
    """Arm 2 — carries ``def name(arg1, ...):``; the example is removed."""

    def __init__(self, task_config: Dict[str, Any]):
        super().__init__({**task_config, "carrier": "signature"})


@register_task("code_summarization_name")
class NameCarrierSummarizationTask(AssertAblationSummarizationTask):
    """Arm 3 — carries the function name only."""

    def __init__(self, task_config: Dict[str, Any]):
        super().__init__({**task_config, "carrier": "name"})


@register_task("code_summarization_bare")
class BareSummarizationTask(AssertAblationSummarizationTask):
    """Arm 4 — carries nothing; the floor."""

    def __init__(self, task_config: Dict[str, Any]):
        super().__init__({**task_config, "carrier": "bare"})
