"""Score the assert-line ablation runs and emit the submission's summary.csv.

Metrics come from the framework's own ``evaloop.analysis.metrics``, not from a
re-derivation here. The headline number is **ASL_pow**, the unweighted
quadratic ASL ``sum_i n_i * i^2 / (T * M)``, because:

* the leaderboard's headline "ASL" column is ``semanticSimilarityScore``, which
  weights each cycle bucket by an LLM judge's similarity score, and the judge
  used for the published data (``gpt-4-turbo-2024-04-09``) is not reachable
  through the course's Vertex credits; and
* ``leaderboard.json``'s ``aslScore`` field is documented as
  ``sum_i i^2 * count_i / 3780`` — exactly ASL_pow at T=378, M=10 — so the
  published number and ours are computed the same way and are comparable.

That comparability doubles as a protocol check: the ``ablation_assert`` arm is
a re-implementation of the standard loop, so its ASL_pow should land near the
published ``aslScore`` for the same model. A large gap means the re-implementation
diverged, and every ablation delta measured against it would be suspect.

Usage:
    python analyze_ablation.py [--results results/ablation] [--data ../assignment2-data]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaloop.analysis import metrics as asl_metrics  # noqa: E402

ARM_ORDER = ["ablation_assert", "ablation_signature", "ablation_name", "ablation_bare"]
ARM_LABEL = {
    "ablation_assert": "assert (name+arity+example)",
    "ablation_signature": "signature (name+arity)",
    "ablation_name": "name only",
    "ablation_bare": "bare (nothing)",
    "code_generation_summarization": "upstream default loop",
}


def load_runs(results_dir: str) -> List[Dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_results.json"))):
        payload = json.load(open(path))
        record = payload[0] if isinstance(payload, list) else payload
        meta = record.get("run_metadata", {})
        cycles = [p["successful_cycles"] for p in record["prompt_results"]]
        max_cycles = record.get("max_cycles") or meta.get("max_cycles") or 10
        runs.append({
            "path": path,
            "model": meta.get("model") or record.get("model", "?"),
            "arm": meta.get("loop_type", "?"),
            "n_tasks": len(cycles),
            "max_cycles": max_cycles,
            "cycles": cycles,
            "cost_usd": meta.get("cost_usd"),
            "token_usage": meta.get("token_usage"),
        })
    return runs


def published_asl(data_dir: str) -> Dict[str, Dict[str, float]]:
    """``model_id -> {successRate, aslScore}`` from the course data package."""
    path = os.path.join(data_dir, "leaderboard.json")
    if not os.path.exists(path):
        return {}
    payload = json.load(open(path))
    rows = payload if isinstance(payload, list) else payload.get(
        "models", payload.get("data", []))
    out = {}
    for row in rows:
        mid = row.get("model_id") or row.get("id")
        if mid:
            out[str(mid)] = {
                "successRate": row.get("successRate"),
                "aslScore": row.get("aslScore"),
            }
    return out


def score(run: Dict) -> Dict:
    cycles, m = run["cycles"], run["max_cycles"]
    n = len(cycles)
    return {
        "success_rate": asl_metrics.pass_at_k(cycles, 1) / n if n else 0.0,
        "asl_pow": asl_metrics.asl_pow(cycles, m),
        "asl_bias": asl_metrics.asl_bias(cycles, m),
        "pass_drop": asl_metrics.pass_drop(cycles, m),
        "histogram": asl_metrics.cycle_histogram(cycles, m),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ablation")
    ap.add_argument("--data", default="../assignment2-data")
    ap.add_argument("--out", default="summary.csv")
    args = ap.parse_args()

    runs = load_runs(args.results)
    if not runs:
        print(f"No *_results.json under {args.results}")
        return
    pub = published_asl(args.data)

    by_model: Dict[str, Dict[str, Dict]] = {}
    for run in runs:
        by_model.setdefault(run["model"], {})[run["arm"]] = {**run, **score(run)}

    total_cost = 0.0
    for model in sorted(by_model):
        arms = by_model[model]
        print(f"\n=== {model} ===")
        reference = pub.get(model, {})
        if reference.get("aslScore") is not None:
            print(f"published baseline (leaderboard aslScore): "
                  f"{reference['aslScore']:.3f}   "
                  f"successRate {reference.get('successRate')}")

        base = arms.get("ablation_assert")
        header = (f"{'arm':30s} {'n':>4} {'pass@1':>7} {'ASL_pow':>8} "
                  f"{'vs assert':>10} {'drop@10':>8} {'cost':>8}")
        print(header)
        print("-" * len(header))
        for arm in ARM_ORDER + [a for a in arms if a not in ARM_ORDER]:
            r = arms.get(arm)
            if r is None:
                continue
            delta = ""
            if base and arm != "ablation_assert" and base["asl_pow"]:
                pct = (r["asl_pow"] - base["asl_pow"]) / base["asl_pow"] * 100
                delta = f"{pct:+.1f}%"
            cost = r["cost_usd"]
            total_cost += cost or 0.0
            print(f"{ARM_LABEL.get(arm, arm):30s} {r['n_tasks']:>4} "
                  f"{r['success_rate']:>6.1%} {r['asl_pow']:>8.3f} "
                  f"{delta:>10} {r['pass_drop']:>7.1%} "
                  f"{('$%.3f' % cost) if cost is not None else 'n/a':>8}")

        if base and reference.get("aslScore") is not None:
            gap = base["asl_pow"] - reference["aslScore"]
            print(f"\nprotocol check: our assert arm {base['asl_pow']:.3f} vs "
                  f"published {reference['aslScore']:.3f}  (gap {gap:+.3f})")

        print("\ncycle histogram (tasks sustaining exactly i cycles):")
        print(f"  {'arm':30s} " + " ".join(f"{i:>4}" for i in range(0, 11)))
        for arm in ARM_ORDER:
            r = arms.get(arm)
            if r:
                print(f"  {ARM_LABEL.get(arm, arm):30s} "
                      + " ".join(f"{c:>4}" for c in r["histogram"]))

    # summary.csv — header fixed by the assignment
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "configuration", "n_tasks",
                         "success_rate", "asl_score"])
        for model in sorted(by_model):
            for arm in ARM_ORDER + [a for a in by_model[model]
                                    if a not in ARM_ORDER]:
                r = by_model[model].get(arm)
                if r:
                    writer.writerow([model, arm, r["n_tasks"],
                                     f"{r['success_rate']:.4f}",
                                     f"{r['asl_pow']:.3f}"])
    print(f"\nwrote {args.out}")
    print(f"total measured API cost across runs: ${total_cost:.3f}")


if __name__ == "__main__":
    main()
