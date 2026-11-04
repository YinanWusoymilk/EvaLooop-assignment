"""Build a stratified worksheet of failed loops for manual failure analysis.

The assignment asks for at least 20 inspected failures and for a comparison
against the baseline's failure modes. Sampling is stratified by arm rather than
drawn uniformly: the whole point of the ablation is that the arms should fail
*differently*, and a uniform sample over all failures would be dominated by
whichever arm fails most.

For every sampled failure the worksheet shows the three things needed to assign
a cause:

1. the original task prompt,
2. the summary produced by the last cycle that passed — i.e. the description the
   model was working from when it failed, and
3. the code generated from that description, with the base/plus test split.

The base/plus split matters: a failure where base passes and only plus fails is
EvalPlus strictness on edge cases, not semantic drift, and should not be counted
as the loop breaking down.

Suggested labels for the `cause` column:

  dropped-requirement   the summary omitted a constraint the tests check
  cannot-implement      the description is faithful; the code is simply wrong
  signature-drift       wrong function name or arity (expected in `bare`)
  lost-example          failure traceable to no longer seeing a worked example:
                        wrong argument order, wrong return container, wrong
                        boundary convention
  extraction            no code block / prose returned instead of code
  plus-only             base tests pass, only the extended tests fail
  other

Usage:
    python make_failure_review.py [--per-arm 6] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from typing import Dict, List, Optional


def load_runs(results_dir: str) -> List[Dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_results.json"))):
        payload = json.load(open(path))
        record = payload[0] if isinstance(payload, list) else payload
        meta = record.get("run_metadata", {})
        runs.append({
            "model": meta.get("model") or record.get("model", "?"),
            "arm": meta.get("loop_type", "?"),
            "max_cycles": record.get("max_cycles") or 10,
            "prompt_results": record["prompt_results"],
        })
    return runs


def failures(run: Dict, min_sustained: int = 1) -> List[Dict]:
    """Loops that broke before the cycle budget ran out.

    ``min_sustained=1`` excludes tasks that failed at cycle 1. Those say
    nothing about the ablation: cycle 1's input is the untouched MBPP Plus
    prompt in every arm, so the carrier has not yet entered the loop and the
    arms are by construction identical there. Including them would fill the
    sample with ordinary one-shot failures (and with EvalPlus edge-case
    strictness, where base passes and only plus fails) instead of the carrier's
    effect. Their count is still reported per arm, and pass@1 being flat across
    arms is the check that the arms really are identical at cycle 1.
    """
    out = []
    for task in run["prompt_results"]:
        sustained = task["successful_cycles"]
        if sustained >= run["max_cycles"] or sustained < min_sustained:
            continue                       # never failed, or failed at cycle 1
        cycles = task.get("cycles", [])
        if not cycles:
            continue
        failing = cycles[-1]               # the loop stops at the first failure
        prior = cycles[-2] if len(cycles) >= 2 else None
        tests = failing.get("test_results", {}) or {}
        out.append({
            "model": run["model"],
            "arm": run["arm"],
            "task_id": task["task_id"],
            "successful_cycles": sustained,
            "initial_prompt": task["initial_prompt"],
            # The description that was fed into the failing cycle: the previous
            # cycle's summary, or the original prompt when it failed at cycle 1.
            "input_description": (prior or {}).get("task_2_output")
                                 or task["initial_prompt"],
            "failing_code": failing.get("task_1_output", ""),
            "base_passed": tests.get("base_success"),
            "plus_passed": tests.get("plus_success"),
        })
    return out


def baseline_failures(data_dir: str, model: str, limit: int) -> List[Dict]:
    """Failed loops from the provided standard-loop records, for comparison."""
    path = os.path.join(data_dir, "loops", f"{model}.json")
    if not os.path.exists(path):
        return []
    payload = json.load(open(path))
    out = []
    for task in payload.get("tasks", []):
        sustained = task.get("successful_cycles", 0)
        if sustained >= 10 or sustained < 1:
            continue
        cycles = task.get("cycles", [])
        if not cycles:
            continue
        failing = cycles[-1]
        prior = cycles[-2] if len(cycles) >= 2 else None
        out.append({
            "model": model,
            "arm": "PROVIDED-baseline",
            "task_id": task["task_id"],
            "successful_cycles": task["successful_cycles"],
            "initial_prompt": task["initial_prompt"],
            "input_description": (prior or {}).get("summary")
                                 or task["initial_prompt"],
            "failing_code": failing.get("code", ""),
            "base_passed": failing.get("base_passed"),
            "plus_passed": failing.get("plus_passed"),
            "boundary_similarity": task.get("boundary_similarity"),
        })
        if len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ablation")
    ap.add_argument("--data", default="../assignment2-data")
    ap.add_argument("--per-arm", type=int, default=6)
    ap.add_argument("--min-sustained", type=int, default=1,
                    help="exclude loops that broke at cycle 1 (see failures())")
    ap.add_argument("--baseline-per-model", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="review")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    runs = load_runs(args.results)
    if not runs:
        print(f"No *_results.json under {args.results}")
        return

    chosen: List[Dict] = []
    print(f"excluding loops that broke at cycle 1 "
          f"(min_sustained={args.min_sustained}): the carrier only enters from "
          f"cycle 2, so those failures carry no information about the arms.\n")
    print(f"{'model':24s} {'arm':22s} {'mid-loop':>9} {'cycle-1':>8} {'sampled':>8}")
    print("-" * 75)
    for run in runs:
        fails = failures(run, args.min_sustained)
        at_cycle1 = sum(1 for t in run["prompt_results"]
                        if t["successful_cycles"] == 0)
        rng.shuffle(fails)
        take = fails[: args.per_arm]
        chosen += take
        print(f"{run['model']:24s} {run['arm']:22s} {len(fails):>9} "
              f"{at_cycle1:>8} {len(take):>8}")

    for model in sorted({r["model"] for r in runs}):
        base = baseline_failures(args.data, model, args.baseline_per_model)
        chosen += base
        if base:
            print(f"{model:24s} {'PROVIDED-baseline':22s} {'-':>9} {'-':>8} {len(base):>8}")

    rng.shuffle(chosen)
    os.makedirs(args.outdir, exist_ok=True)

    csv_path = os.path.join(args.outdir, "failure_review.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "model", "arm", "task_id", "successful_cycles",
                         "base_passed", "plus_passed", "cause", "notes"])
        for i, f in enumerate(chosen, 1):
            writer.writerow([i, f["model"], f["arm"], f["task_id"],
                             f["successful_cycles"], f["base_passed"],
                             f["plus_passed"], "", ""])

    md_path = os.path.join(args.outdir, "failure_review.md")
    with open(md_path, "w") as fh:
        fh.write("# Failed-loop review worksheet\n\n")
        fh.write(f"{len(chosen)} failures, stratified by arm "
                 f"(seed {args.seed}). Fill `cause` and `notes` in "
                 "`failure_review.csv`.\n\n")
        fh.write("Labels: dropped-requirement | cannot-implement | "
                 "signature-drift | lost-example | extraction | plus-only | "
                 "other\n\n---\n\n")
        for i, f in enumerate(chosen, 1):
            fh.write(f"## {i}. `{f['arm']}` — {f['task_id']} "
                     f"({f['model']}), broke after "
                     f"{f['successful_cycles']} cycle(s)\n\n")
            fh.write(f"tests: base={f['base_passed']} plus={f['plus_passed']}")
            if f.get("boundary_similarity") is not None:
                fh.write(f" · judge boundary_similarity={f['boundary_similarity']}")
            fh.write("\n\n**Original task**\n\n```\n"
                     f"{f['initial_prompt'].strip()}\n```\n\n")
            fh.write("**Description fed into the failing cycle**\n\n```\n"
                     f"{f['input_description'].strip()}\n```\n\n")
            fh.write("**Code generated from it (failed)**\n\n```python\n"
                     f"{f['failing_code'].strip()}\n```\n\n---\n\n")

    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
