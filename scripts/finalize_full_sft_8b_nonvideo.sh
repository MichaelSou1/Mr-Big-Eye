#!/usr/bin/env bash
# CPU-only finalization gate for the 8B dense/full-SFT non-video controls.
# Run this after the TabMWP resume script finishes.
set -euo pipefail

cd /home/gpus/Mr-Big-Eye-internalization

PY=${PY:-/home/gpus/anaconda3/envs/mbe-up/bin/python}
LOG=${LOG:-/home/gpus/logs/full_sft_8b_nonvideo}
mkdir -p "$LOG" docs/reviews data/distill/poc

OUT="$LOG/finalize_$(date '+%Y%m%d_%H%M%S').log"

say() {
  echo "[$(date '+%F %T')] $*" | tee -a "$OUT"
}

say "Finalizing Full-SFT 8B non-video evidence (CPU-only)."

active=$(
  ps -eo pid,ppid,stat,etime,cmd \
    | awk '/scripts\/(battery_n400\.py|run_full_sft_8b_nonvideo\.sh|eval_full_sft_n400\.py|poc_sft_full_8b\.py)/ {print}' \
    || true
)
if [[ -n "$active" ]]; then
  say "Refusing to finalize while related experiment processes are active:"
  echo "$active" | tee -a "$OUT"
  exit 5
fi

say "Exporting TabMWP Full-SFT posthoc answer classification."
"$PY" scripts/summarize_full8b_tabmwp_posthoc.py 2>&1 | tee -a "$OUT"

say "Running strict audit."
"$PY" scripts/audit_full_sft_8b_nonvideo.py --strict \
  --out-json data/distill/poc/full_sft_8b_nonvideo_audit.json \
  --out-md docs/reviews/full_sft_8b_nonvideo_audit.md \
  2>&1 | tee -a "$OUT"

say "Exporting paper-facing evidence tables and resume manifest."
"$PY" scripts/export_full_sft_8b_nonvideo_evidence.py 2>&1 | tee -a "$OUT"

say "Final strict audit summary:"
python - <<'PY' | tee -a "$OUT"
import json
from pathlib import Path
o=json.loads(Path('data/distill/poc/full_sft_8b_nonvideo_audit.json').read_text())
print('complete:', o['complete'])
print('missing:', o['missing_requirements'])
print('required:')
for k,v in o['required'].items():
    print(f'  {k}: {v}')
print('checkpoint:', o['checkpoint'])
PY

say "Full-SFT 8B non-video finalization passed."
