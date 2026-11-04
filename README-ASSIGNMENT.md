# Assignment 4 — Assert-Line Ablation for EvaLoop

CSC 591/791 — Trustworthy AI for Software Engineering.

**Public fork:** <https://github.com/YinanWusoymilk/EvaLooop-assignment> (fork
of <https://github.com/evalooop/EvaLooop>), branch `dev` — the work is also on
the branch `assert-line-ablation`, at the same commit. The complete diff against
upstream (`evalooop/EvaLooop@ce22fe2`) is in `changes.patch`.

EvaLoop re-attaches the original prompt's final assert line to every summary so
that the function signature survives the loop. That line is also a worked
example — one input with its expected output, re-injected from outside the loop
every cycle. This extension adds a loop type that holds the standard loop fixed
and weakens only that carrier, in four arms:

| arm | carrier appended to the summary | information retained |
| --- | --- | --- |
| `ablation_assert` | `assert f(3, 4) == 7` | name, arity, example |
| `ablation_signature` | `def f(arg1, arg2):` | name, arity |
| `ablation_name` | ``The function must be named `f`.`` | name |
| `ablation_bare` | *(nothing)* | nothing |

`ablation_assert` re-implements the standard loop and is the baseline the other
arms are measured against.

---

## What I added, and what I changed in the framework

**New files** (the extension proper):

| file | purpose |
| --- | --- |
| `evaloop/tasks/assert_ablation.py` | the four arms; `signature_from_assert()` |
| `evaloop/models/gemini_vertex.py` | Gemini backend on Vertex AI (ADC), batched, rate-limited, with cost accounting |
| `run_ablation.sh` | runs all 8 (model × arm) configurations |
| `analyze_ablation.py` | scores the runs, writes `summary.csv` |
| `make_failure_review.py` | builds the stratified failed-loop worksheet |
| `progress.sh` | live progress/rate of a run |

**Modified files**, and why each change was necessary:

| file | change | why |
| --- | --- | --- |
| `evaloop/tasks/__init__.py` | import the new module | `@register_task` only fires on import |
| `evaloop/core/config.py` | `loop_type` and `max_tasks` fields; arms added to `get_task_configs`; Gemini routing in `_create_model_config`; Vertex settings | `get_task_configs` is the only place a loop's composition is defined and it raised on anything but the default. Gemini ids previously fell through to the HuggingFace branch and the run died trying to download them from the Hub |
| `evaloop/core/evaluator.py` | use `config.loop_type`; honour `max_tasks`; tag result files with the arm; write a `run_metadata` block | the loop type was hard-coded; all four arms would otherwise overwrite one result file; every reported number needs its settings and measured cost attached |
| `evaloop/models/factory.py` | register `gemini`; make the HuggingFace/vLLM imports lazy | importing them eagerly made `pip install torch` (~2.5 GB) a prerequisite for evaluating a hosted model |
| `evaloop/cli.py` | `--loop_type`, `--max_tasks`, `--vertex_project`, `--vertex_location`, `--api_workers` | `fire` exposes keyword args as flags |

Upstream behaviour is unchanged: `--loop_type code_generation_summarization`
(the default) still runs the original loop, and the repository's own test suite
passes (75 passed, 1 skipped).

---

## Dependencies

Python 3.11+. Stages that touch no model need nothing but the repo.

```bash
python3 -m venv .venv
./.venv/bin/pip install fire openai pyyaml pandas tqdm evalplus google-genai pytest
```

`torch`, `transformers` and `vllm` are **not** needed for API-only runs (see the
lazy-import change above).

## Credentials

Vertex AI via Application Default Credentials — no API key:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <PROJECT_ID>
export GOOGLE_CLOUD_PROJECT=<PROJECT_ID>
```

`make_client()` falls back to `GOOGLE_API_KEY` (AI Studio) if that variable is
set, so either backend works.

## Running

One configuration:

```bash
EVALPLUS_MAX_MEMORY_BYTES=-1 ./.venv/bin/python -m evaloop.cli evaluate --model gemini-2.5-flash --loop_type ablation_signature --max_tasks 120 --max_cycles 10
```

All eight, as reported:

```bash
./run_ablation.sh 120
```

Then score them and write `summary.csv`:

```bash
./.venv/bin/python analyze_ablation.py
```

And build the failed-loop worksheet:

```bash
./.venv/bin/python make_failure_review.py --per-arm 4 --baseline-per-model 3
```

### Two environment requirements

**`EVALPLUS_MAX_MEMORY_BYTES=-1` is required on macOS.** EvalPlus sets
`RLIMIT_AS` unconditionally; Darwin rejects it (`ValueError: current limit
exceeds maximum limit`), every functional test fails, and every ASL reads 0.
`-1` is EvalPlus's own documented escape hatch and disables only the memory
cap — the subprocess isolation, the per-test timeout and the destructive-call
guard all remain. `run_ablation.sh` exports it.

**Thinking is disabled** (`disable_thinking`, default on). Gemini 2.5+ count
reasoning tokens against `max_output_tokens`; at EvaLoop's 1024-token budget
this truncated 35 of 120 cycle-1 generations mid-expression and dropped pass@1
from 80.0% to 62.5%. The protocol specifies 1024 tokens for the answer, so
reasoning is turned off to make the budget mean that.

---

## Models and settings

| | |
| --- | --- |
| Models | `gemini-2.5-flash`, `gemini-3.5-flash-lite` (Vertex AI, location `global`) |
| Decoding | temperature 0.0, top-p 1.0, 1024 new tokens, thinking disabled |
| Cycles | max 10; one run per configuration |
| Tasks | first 120 of EvalPlus's MBPP Plus ordering (`--max_tasks 120`) |
| Concurrency | 12 workers, AIMD rate limiter, 12 retries |
| Measured cost | **$3.08** total — 14,342 calls, 1.54M input / 1.05M output tokens, $0.0032 per task-arm |

No LLM is used as a judge. The headline metric is `ASL_pow`
($\sum_i n_i i^2 / (TM)$) rather than the leaderboard's semantic ASL, because
the published judge (`gpt-4-turbo-2024-04-09`) is not reachable through Vertex;
`leaderboard.json`'s `aslScore` field is documented as the same formula, so the
published number and mine are computed alike.

---

## Reproducing each reported number

| number | where it comes from |
| --- | --- |
| ASL, pass@1, cycle histograms, per-arm cost | `analyze_ablation.py` over `results/ablation/*.json`; also `summary.csv` |
| per-run token usage and cost | `run_metadata.token_usage` / `run_metadata.cost_usd` inside each result file, accumulated from the API's own `usage_metadata` |
| `hard_failures = 0` for all 8 runs | `run_metadata.token_usage.hard_failures`; `run_ablation.sh` also prints it per run and flags any run as untrustworthy |
| mid-loop vs cycle-1 failure counts | the table printed by `make_failure_review.py` |
| failure-cause breakdown (35 loops) | `review/failure_review.csv`, column `cause`; the evidence for each is in `review/failure_review.md` |
| protocol check (8.115 vs 6.923 / 7.167) | `analyze_ablation.py` prints it against `assignment2-data/leaderboard.json`; the 120-task figure is the same metric recomputed over the first 120 tasks of `assignment2-data/loops/gemini-2.5-flash.json` |
| `signature_from_assert` 378/378 | re-run the extractor over every `initial_prompt` in `assignment2-data/loops/*.json` and compare the recovered name with the cycle-1 code's `def` |

Set `--seed` on `make_failure_review.py` to reproduce the same sample
(the reported one is seed 7).

## Provided data package

Used as a cross-check only, never copied into the submission; records are
referenced by `model_id`, `task_id` and cycle. Note that of the three baseline
failures sampled, all three store `Error generating response: 'NoneType' object
is not subscriptable` where the generated code belongs — API errors recorded as
loop failures, each scored `boundary_similarity = 1.0` by the published judge.

## Safety

Generated code is untrusted model output and is executed only through
EvalPlus's harness, which runs each candidate in a separate process with a
timeout and with destructive os/shutil/subprocess functions disabled. The
memory cap is off on macOS for the reason given above; no generated code is run
outside that harness.
