#!/usr/bin/env bash
# Show live progress of the ablation runs.
#
# Rate is derived from the per-arm logs, not from the process table: this
# machine's sandboxing makes `pgrep` unreliable here, whereas a log that is
# still growing is direct evidence the run is alive.
#
# Usage:  ./progress.sh

set -uo pipefail
cd "$(dirname "$0")"

OUT="results/ablation"
LOGS="$OUT/logs"

echo "=== script progress ==="
if [[ -f results_run.log ]]; then
  grep -E "^\[|done in|FAILED|all runs finished|NOT TRUSTWORTHY" results_run.log || true
else
  echo "  (no results_run.log yet)"
fi

echo
echo "=== per-arm ==="
printf "  %-44s %-9s %7s %7s %6s %8s\n" ARM CYCLE CALLS 429 GAVEUP RATE/s
for f in "$LOGS"/*.log; do
  [[ -f "$f" ]] || continue
  name=$(basename "$f" .log)
  start=$(head -1 "$f" | grep -o '2026-[0-9-]* [0-9:]*' || true)
  calls=$(grep -c '200 OK' "$f" || true)
  rl=$(grep -c '429' "$f" || true)
  gave=$(grep -c 'gave up after' "$f" || true)
  cyc=$(grep -o 'Starting cycle [0-9]*/[0-9]*' "$f" | tail -1 | awk '{print $3}')
  # Finished runs: measure to their last log line, not to now.
  last=$(tail -1 "$f" | grep -o '2026-[0-9-]* [0-9:]*' || true)
  rate=$(python3 -c "
import datetime,sys
s='$start'; l='$last'; n=$calls
if not s or not n: print('-'); sys.exit()
st=datetime.datetime.strptime(s,'%Y-%m-%d %H:%M:%S')
en=datetime.datetime.strptime(l,'%Y-%m-%d %H:%M:%S') if l else datetime.datetime.now()
el=max((en-st).total_seconds(),1)
print(f'{n/el:.2f}')
" 2>/dev/null || echo '-')
  printf "  %-44s %-9s %7s %7s %6s %8s\n" "$name" "${cyc:-?}" "$calls" "$rl" "$gave" "$rate"
done

echo
echo "=== finished result files ==="
ls -1 "$OUT"/*.json 2>/dev/null | sed 's|.*/|  |' || echo "  (none yet)"
