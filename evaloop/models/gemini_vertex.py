"""Gemini backend on Vertex AI, authenticated with Application Default Credentials.

Why a native backend instead of reusing :class:`OpenAILLM` with a ``base_url``:
Vertex does expose an OpenAI-compatible endpoint, but it authenticates with a
``gcloud`` OAuth access token that expires after about an hour. A full run here
is 378 tasks x up to 10 cycles x 2 generations, which takes considerably longer
than that, so a static token would strand the run part-way through and waste
the spend. ``google-genai`` with ``vertexai=True`` refreshes ADC itself, and it
also reports ``usage_metadata`` per call, which is what the report's cost table
is built from.

This class implements ``batch_generate``, which
:class:`evaloop.evaluation.cycle.BatchEvaluationCycle` detects and prefers: the
tasks within one cycle are independent, so they are issued concurrently instead
of one at a time.

Setup::

    gcloud auth application-default login
    gcloud auth application-default set-quota-project <PROJECT_ID>
"""

import logging
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .base import BaseLLM

logger = logging.getLogger(__name__)

#: USD per 1M tokens, Vertex AI / Gemini API list prices.
PRICING = {
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "gemini-2.5-pro": {"in": 1.25, "out": 10.00},
    "gemini-3.5-flash-lite": {"in": 0.30, "out": 2.50},
    "gemini-3.5-flash": {"in": 1.50, "out": 9.00},
    "gemini-3.7-flash": {"in": 0.75, "out": 3.75},
    "gemini-3.8-flash": {"in": 0.75, "out": 3.75},
    "gemini-2.0-flash-lite": {"in": 0.075, "out": 0.30},
    "gemini-2.0-flash": {"in": 0.10, "out": 0.40},
}

TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500",
             "INTERNAL", "DEADLINE", "ABORTED")


class _AdaptiveLimiter:
    """Paces request *starts* so the quota is approached rather than hit.

    Retry-after-failure alone performs badly against Vertex's rolling
    per-minute budget: twelve workers burst, all collide, and each then sleeps
    out a randomised backoff — observed stalling the whole process for 30s at a
    time while quota sat unused, with throughput oscillating between 5/s and
    1/s. Pacing the starts keeps the request stream smooth instead.

    The target rate is not known in advance and varies with what else is using
    the project, so it is learned: multiplicative decrease on a rejection,
    gentle additive-style recovery on success (AIMD).
    """

    def __init__(self, rps: float = 6.0, min_rps: float = 0.5,
                 max_rps: float = 25.0):
        self._interval = 1.0 / rps
        self._floor = 1.0 / max_rps
        self._ceil = 1.0 / min_rps
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def penalize(self) -> None:
        with self._lock:
            self._interval = min(self._interval * 1.6, self._ceil)

    def reward(self) -> None:
        with self._lock:
            self._interval = max(self._interval * 0.97, self._floor)

    @property
    def rps(self) -> float:
        with self._lock:
            return 1.0 / self._interval


class GeminiVertexLLM(BaseLLM):
    """Gemini via Vertex AI.

    Recognised ``model_config`` keys: ``model_id``, ``project``, ``location``,
    ``temperature``, ``top_p``, ``max_tokens``, ``system_prompt``, ``workers``.
    """

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        from google import genai  # imported lazily: optional dependency

        self.model_id = model_config.get("model_id", "gemini-2.5-flash-lite")
        self.temperature = model_config.get("temperature", 0.0)
        self.top_p = model_config.get("top_p", 1.0)
        self.max_tokens = model_config.get("max_tokens", 1024)
        self.system_prompt = model_config.get("system_prompt") or None
        self.disable_thinking = bool(model_config.get("disable_thinking", True))
        self.workers = int(model_config.get("workers", 16))
        # Vertex returns 429 readily on the shared publisher endpoints: it is a
        # requests-per-minute limit, not a concurrency limit, so lowering
        # `workers` does not avoid it — only backing off does. Measured on this
        # project, a single process at 8-16 workers with 8 retries completes
        # 60/60 calls; four concurrent processes at 12 workers with 6 retries
        # exhausted retries on ~1% of calls. Those were silently recorded as
        # empty generations, i.e. as loop failures, which biases ASL downward.
        # Hence: retry hard, and count what still fails (see hard_failures).
        self.max_retries = int(model_config.get("max_retries", 12))

        project = (model_config.get("project")
                   or os.environ.get("GOOGLE_CLOUD_PROJECT"))
        location = (model_config.get("location")
                    or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global")
        if not project:
            raise ValueError(
                "No Google Cloud project. Pass project in the model config or "
                "set GOOGLE_CLOUD_PROJECT, then run:\n"
                "  gcloud auth application-default login\n"
                "  gcloud auth application-default set-quota-project <PROJECT_ID>"
            )
        self.client = genai.Client(vertexai=True, project=project,
                                   location=location)

        # Measured token usage, for the report's cost table.
        self._usage = Counter()
        self._usage_lock = threading.Lock()
        self._limiter = _AdaptiveLimiter(
            rps=float(model_config.get("target_rps", 6.0)))
        logger.info("GeminiVertexLLM ready: model=%s project=%s location=%s",
                    self.model_id, project, location)

    # -- generation ------------------------------------------------------

    def _config(self):
        from google.genai import types
        cfg = dict(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
            system_instruction=self.system_prompt,
        )
        if self.disable_thinking:
            # Gemini 2.5+ emit reasoning tokens that count against
            # max_output_tokens. With EvaLoop's 1024-token budget this starves
            # the answer: measured on gemini-2.5-flash, 35 of 120 cycle-1
            # generations were cut off mid-expression ("for i in range(1, n"),
            # dropping pass@1 from the published 80.0% to 62.5% on the same 120
            # tasks — an artifact of the budget, not of the model or the
            # ablation. The protocol specifies 1024 *new tokens per
            # generation*, i.e. for the answer, so thinking is disabled to make
            # the budget mean that. Side effects: ~5x fewer output tokens,
            # proportionally cheaper and faster.
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**cfg)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate one completion, retrying transient API failures.

        Returns the text, or ``""`` if the call could not be completed. An
        empty string propagates into the loop as a failed cycle rather than an
        exception, which is how the upstream backends behave — but it is
        indistinguishable from the model genuinely producing nothing, so every
        such case is counted in :attr:`hard_failures`. A run with a non-zero
        count has infrastructure failures recorded as model failures and its
        ASL is not trustworthy.
        """
        cfg = self._config()
        for attempt in range(self.max_retries):
            self._limiter.acquire()
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id, contents=prompt, config=cfg)
                self._record(resp)
                self._limiter.reward()
                text = (resp.text or "").strip()
                if not text:
                    with self._usage_lock:
                        self._usage["empty_responses"] += 1
                return text
            except Exception as exc:                             # noqa: BLE001
                msg = str(exc)
                throttled = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                if throttled:
                    self._limiter.penalize()
                    with self._usage_lock:
                        self._usage["throttled"] += 1
                if (not any(code in msg for code in TRANSIENT)
                        or attempt == self.max_retries - 1):
                    logger.error("Gemini call gave up after %d attempts (%s): %s",
                                 attempt + 1, self.model_id, msg[:200])
                    with self._usage_lock:
                        self._usage["hard_failures"] += 1
                    return ""
                # Short jittered backoff only: the limiter now does the pacing,
                # so a long sleep here would idle a worker while quota frees up.
                time.sleep(random.uniform(0, min(2 ** attempt, 8)))
        with self._usage_lock:
            self._usage["hard_failures"] += 1
        return ""

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate for every prompt concurrently, preserving input order."""
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0])]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(prompts))) as pool:
            return list(pool.map(self.generate, prompts))

    # -- accounting ------------------------------------------------------

    def _record(self, resp) -> None:
        usage = getattr(resp, "usage_metadata", None)
        if usage is None:
            return
        tokens_in = getattr(usage, "prompt_token_count", 0) or 0
        tokens_out = ((getattr(usage, "candidates_token_count", 0) or 0)
                      + (getattr(usage, "thoughts_token_count", 0) or 0))
        with self._usage_lock:
            self._usage["in"] += tokens_in
            self._usage["out"] += tokens_out
            self._usage["calls"] += 1

    @property
    def usage(self) -> Dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)

    def cost_usd(self) -> Optional[float]:
        """Measured spend, or ``None`` if this model has no price entry."""
        price = PRICING.get(self.model_id)
        if price is None:
            return None
        u = self.usage
        return u.get("in", 0) / 1e6 * price["in"] + u.get("out", 0) / 1e6 * price["out"]

    @property
    def hard_failures(self) -> int:
        """Calls that never completed. Must be 0 for a run to be usable."""
        return self.usage.get("hard_failures", 0)

    def usage_report(self) -> str:
        u = self.usage
        cost = self.cost_usd()
        line = (f"{self.model_id}: {u.get('calls', 0)} calls, "
                f"{u.get('in', 0):,} in / {u.get('out', 0):,} out tokens")
        line += f", ${cost:.4f}" if cost is not None else ", cost n/a"
        hard = u.get("hard_failures", 0)
        empty = u.get("empty_responses", 0)
        line += (f", throttled={u.get('throttled', 0)}"
                 f", settled_rps={self._limiter.rps:.1f}"
                 f", hard_failures={hard}, empty_responses={empty}")
        if hard:
            line += "  <-- RUN NOT TRUSTWORTHY"
        return line

    def __str__(self) -> str:
        return f"GeminiVertexLLM(model_name={self.model_id})"
