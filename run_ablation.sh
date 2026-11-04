#!/usr/bin/env bash
# Run the assert-line ablation: 2 models x 4 arms on the full MBPP Plus set.
#
# Runs are SEQUENTIAL, one process at a time. An earlier version ran the four
# arms of a model concurrently; Vertex's 429 limit on the shared publisher
# endpoints is a requests-per-minute limit, so four processes x 12 workers
# exhausted the retry budget on ~1% of calls. Those calls returned an empty
# generation, which the loop records as a failed cycle — infrastructure noise
# masquerading as model failure, biasing ASL downward and unevenly across arms.
# Measured on this project: one process at 12 workers with 12 retries completes
# every call. Sequential is slower in wall-clock and correct.
#
# EVALPLUS_MAX_MEMORY_BYTES=-1 is required on macOS: EvalPlus sets RLIMIT_AS
# unconditionally and Darwin rejects it ("current limit exceeds maximum limit"),
# which makes every functional test fail and every ASL read 0. -1 is EvalPlus's
# own documented escape hatch and disables only the memory cap; subprocess
# isolation, the timeout and the destructive-call guard all remain.
#
# Usage:  ./run_ablation.sh [N_TASKS]

set -uo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-csc-591-assignment1}"
MODELS=("gemini-2.5-flash" "gemini-3.5-flash-lite")
ARMS=("ablation_assert" "ablation_signature" "ablation_name" "ablation_bare")
MAX_CYCLES=10
API_WORKERS="${API_WORKERS:-12}"
# Regional endpoint rather than "global". Measured A/B was inconclusive --
# throughput is governed by a rolling per-project request budget, so
# back-to-back tests interfere with each other -- but a regional endpoint may
# draw on a separate quota pool, and it costs nothing to prefer one.
VERTEX_LOCATION="${VERTEX_LOCATION:-global}"
OUT="results/ablation"
LOGS="$OUT/logs"

# A plain string, not an array: macOS ships bash 3.2, where expanding an
# empty array under `set -u` aborts with "unbound variable".
N_TASKS_ARG=""
if [[ $# -ge 1 ]]; then
  N_TASKS_ARG="--max_tasks $1"
  echo "Restricting to the first $1 tasks"
fi

export GOOGLE_CLOUD_PROJECT="$PROJECT"
export EVALPLUS_MAX_MEMORY_BYTES=-1

mkdir -p "$LOGS"
PY=./.venv/bin/python

any_fail=0
total=$(( ${#MODELS[@]} * ${#ARMS[@]} ))
i=0
start_all=$(date +%s)

echo "project=$PROJECT  location=$VERTEX_LOCATION  workers=$API_WORKERS  max_cycles=$MAX_CYCLES  runs=$total"
echo

for model in "${MODELS[@]}"; do
  for arm in "${ARMS[@]}"; do
    i=$(( i + 1 ))
    name="${model}_${arm}"
    echo "[$i/$total] $name  ($(date +%H:%M:%S))"
    t0=$(date +%s)

    $PY -m evaloop.cli evaluate \
      --model "$model" \
      --loop_type "$arm" \
      --max_cycles "$MAX_CYCLES" \
      --api_workers "$API_WORKERS" \
      --vertex_location "$VERTEX_LOCATION" \
      --temperature 0.0 \
      --top_p 1.0 \
      --output_dir "$OUT" \
      --experiment_name "$model" \
      $N_TASKS_ARG \
      > "$LOGS/${name}.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))

    if [[ $rc -ne 0 ]]; then
      echo "    FAILED (exit $rc) after ${dt}s — see $LOGS/${name}.log"
      any_fail=$(( any_fail + 1 ))
      continue
    fi

    # A run is only usable if no API call was abandoned: an abandoned call is
    # recorded as an empty generation, i.e. as a broken loop.
    usage_line=$(grep -o "API usage — .*" "$LOGS/${name}.log" | tail -1)
    echo "    done in ${dt}s | ${usage_line:-no usage line}"
    if echo "$usage_line" | grep -q "RUN NOT TRUSTWORTHY"; then
      echo "    ^^ hard failures present; this run must be repeated"
      any_fail=$(( any_fail + 1 ))
    fi
  done
done

echo
echo "=== all runs finished in $(( ($(date +%s) - start_all) / 60 )) min (problem runs: $any_fail) ==="
ls -la "$OUT"/*.json 2>/dev/null
exit $(( any_fail > 0 ? 1 : 0 ))
