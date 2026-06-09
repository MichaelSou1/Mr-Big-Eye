#!/usr/bin/env bash
# Stop the distillation model servers started by serve_vlm_4b.sh / serve_text_30b.sh.
set -uo pipefail
for pid_file in /home/gpus/logs/serve-vlm-4b.pid /home/gpus/logs/serve-text-30b.pid; do
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "stopping $pid ($pid_file)"; kill "$pid" 2>/dev/null
      # sglang spawns child workers; kill the process group too.
      pkill -P "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
done
echo "done"
