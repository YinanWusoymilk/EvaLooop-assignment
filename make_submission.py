"""Assemble the Assignment 4 submission ZIP.

Contents follow the assignment's list: the report, the README (carrying the
fork link), a patch against upstream, the framework's results JSON for every
(model, configuration) run, and summary.csv. The adjudication worksheet is
included as evidence for the failure analysis.

The provided data package is deliberately NOT copied in — the assignment says
to reference its records by model_id / task_id / cycle instead.

Usage:
    python make_submission.py
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile

OUT = "submission"
ZIP = "assignment4_assert_ablation.zip"

# (source, path inside the zip)
FILES = [
    ("report.pdf", "report.pdf"),
    ("report.tex", "report.tex"),
    ("README-ASSIGNMENT.md", "README.md"),
    ("changes.patch", "changes.patch"),
    ("summary.csv", "summary.csv"),
    ("review/failure_review.csv", "failure_analysis/failure_review.csv"),
    ("review/failure_review.md", "failure_analysis/failure_review.md"),
    # The extension's own code, for reading without applying the patch.
    ("evaloop/tasks/assert_ablation.py", "code/evaloop/tasks/assert_ablation.py"),
    ("evaloop/models/gemini_vertex.py", "code/evaloop/models/gemini_vertex.py"),
    ("run_ablation.sh", "code/run_ablation.sh"),
    ("analyze_ablation.py", "code/analyze_ablation.py"),
    ("make_failure_review.py", "code/make_failure_review.py"),
    ("make_submission.py", "code/make_submission.py"),
    ("progress.sh", "code/progress.sh"),
]

RESULTS_DIR = "results/ablation"


def main() -> None:
    if os.path.exists(OUT):
        shutil.rmtree(OUT)

    missing = []
    for src, dst in FILES:
        if not os.path.exists(src):
            missing.append(src)
            continue
        target = os.path.join(OUT, dst)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(src, target)

    # Result JSONs, one per (model, configuration).
    runs = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith("_results.json"))
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    manifest = []
    for f in runs:
        shutil.copy2(os.path.join(RESULTS_DIR, f), os.path.join(OUT, "results", f))
        payload = json.load(open(os.path.join(RESULTS_DIR, f)))
        rec = payload[0] if isinstance(payload, list) else payload
        meta = rec.get("run_metadata", {})
        usage = meta.get("token_usage", {})
        manifest.append({
            "file": f,
            "model": meta.get("model"),
            "configuration": meta.get("loop_type"),
            "n_tasks": meta.get("n_tasks"),
            "max_cycles": meta.get("max_cycles"),
            "temperature": meta.get("temperature"),
            "calls": usage.get("calls"),
            "hard_failures": usage.get("hard_failures"),
            "cost_usd": meta.get("cost_usd"),
        })

    # A run with abandoned API calls has infrastructure failures recorded as
    # loop failures; refuse to ship one silently.
    bad = [m for m in manifest if (m["hard_failures"] or 0) > 0]
    if bad:
        print("REFUSING TO BUILD — runs with hard_failures > 0:")
        for m in bad:
            print(f"  {m['file']}: hard_failures={m['hard_failures']}")
        raise SystemExit(1)

    with open(os.path.join(OUT, "results", "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    if missing:
        print("missing (not packaged):")
        for m in missing:
            print(f"  {m}")

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for f in sorted(files):
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, OUT))

    print(f"\nbuilt {ZIP}  ({os.path.getsize(ZIP)/1e6:.1f} MB)")
    with zipfile.ZipFile(ZIP) as z:
        for n in sorted(z.namelist()):
            print(f"  {n:52s} {z.getinfo(n).file_size:>9,} B")
    print(f"\n{len(runs)} runs packaged, all with hard_failures = 0")


if __name__ == "__main__":
    main()
