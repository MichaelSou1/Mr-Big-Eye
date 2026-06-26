# TabMWP Full-SFT Battery Resume Manifest

- CPU-only readiness precheck: `PRECHECK_ONLY=1 REQUIRE_GPU_IDLE=0 bash scripts/resume_full_sft_8b_tabmwp_battery.sh`
- Precheck command: `PRECHECK_ONLY=1 bash scripts/resume_full_sft_8b_tabmwp_battery.sh`
- GPU command: `bash scripts/resume_full_sft_8b_tabmwp_battery.sh`
- Finalize command: `bash scripts/finalize_full_sft_8b_nonvideo.sh`
- Strict audit: `/home/gpus/anaconda3/envs/mbe-up/bin/python scripts/audit_full_sft_8b_nonvideo.py --strict`
- Posthoc command: `/home/gpus/anaconda3/envs/mbe-up/bin/python scripts/summarize_full8b_tabmwp_posthoc.py`
- Posthoc strict check: `/home/gpus/anaconda3/envs/mbe-up/bin/python scripts/summarize_full8b_tabmwp_posthoc.py --strict`
- Resource guard: GPU idle required by default; max used 2048MB/GPU, min free 16000MB/GPU, min disk 40GB; orchestrator host must include `xiaomimimo.com`
- Required interventions for completed outputs: `corrupt,delete,filler,paraphrase,shuffle,truncate` plus `details[].answers`
- Missing Mimo paraphrases: 0
- Mimo cache entries: 387
- Old cache entries: 387
- Base-CoT cache entries: 400/400
- Common base_md5 mismatches: 0
- TabMWP weight exists: False (None GB)

## Expected Outputs

- `data/distill/poc/battery_full8b_tabmwp_present.json`
- `data/distill/poc/battery_full8b_tabmwp_masked.json`

## Posthoc Outputs

- `data/distill/poc/full8b_tabmwp_battery_posthoc.json`
- `docs/reviews/full8b_tabmwp_battery_posthoc.md`

## Missing Mimo Keys


## Notes

- Both expected TabMWP full-battery outputs exist and contain all six interventions plus details[].answers.
- The resume script removed the large TabMWP weight shard after both expected outputs passed readiness checks.
- The strict audit and posthoc strict check pass; no further GPU run is required for the non-video 8B Full-SFT control.
- The old non-Mimo cache is complete and base_md5-compatible for common entries, but the completed resume path used the Mimo cache.
- Base-CoT cache hits are bound to a lightweight checkpoint fingerprint; stale or old-format cache entries are regenerated instead of reused.
- The finalize command also runs the posthoc command to classify shuffle/filler/paraphrase answers from details[].answers.
