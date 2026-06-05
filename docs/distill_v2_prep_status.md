# Distill v2 (hard rebalance) — prep status & run book

**Date:** 2026-06-03 · **Branch:** distillation · **Spec:** [distill_hard_rebalance_spec.md](distill_hard_rebalance_spec.md)

All **non-GPU preparation is done**. Ingest / teacher / train are staged but
**blocked on GPU** (all 4 cards in use at prep time). Decisions taken:
**all-medium (skip long)** + **write to a v2-isolated dir** (global eval files untouched).

## What changed (code, additive / backward-compatible)
- `scripts/sample_videomme.py` — `--modality-weighted` mode: greedy modality-target
  selection (over-weights `joint`), held-out carve-out, writes train/heldout/ingest lists.
- `scripts/build_videomme_eval.py` — `--sampled-file` / `--out-dir` (build cases into v2 dir).
- `scripts/distill_build_dataset.py` — `--max-train-samples` cap (drops whole videos, never splits a trajectory) to honor C1.
- `scripts/ingest_videomme.py` — `MBE_MANIFEST` env override (target the v2 manifest).
- `scripts/run_teacher_4gpu.sh` — `SHARD_DIR` env override (don't clobber the scaleup shards/caches).

## Selection (deterministic, seed 42)
| set | videos | train_samples (proj) | joint share |
|---|---|---|---|
| reuse medium (pilot, has trajectories) | 49 | — | 14% |
| **new train** medium (joint-weighted) | 120 | — | 27% |
| **TRAIN combined** | **169** | **≈1538** (9.1/video) | **23%** |
| heldout seed (already ingested, 48 cases preserved) | 16 | — | 10% |
| new heldout | 14 | — | 50% |
| **HELDOUT combined** | **30 (90 cases)** | eval-only | **29%** |
| **new videos to ingest** | **134** (120 train + 14 heldout) | | |

**Note (data-limited):** spec targeted `joint ≥35%`; medium-only Video-MME caps the
achievable mix — new-train hit 27% joint, combined-train 23% (reuse-49 anchors it at
14%). Long videos were skipped because Video-MME's long bucket is 30–60 min (only 1
video ≤30 min), which §3.1's "≤30 min cap" can't satisfy.

The 48 original `hardeval_medium.jsonl` cases are a **verified subset** of the 90-case
heldout (identical question/video ids), so base=0.562 / pilot=0.708 stay valid references.

## Artifacts written
- `data/videomme_src/v2_train.json` (120), `v2_heldout_new.json` (14), `v2_to_ingest.json` (134), `v2_cases.json` (150), `v2_sampled.json` (199)
- `eval/audiovisual/v2/questions.jsonl` (450 cases / 150 videos) + `video_manifest.json`
- `data/distillation/v2/teacher_cases.jsonl` (360), `heldout_cases.jsonl` (90)
- `data/distillation/v2/teacher_shards/cases_{0..3}.jsonl` (90 each)

## Run book — when a GPU frees (run from repo root)
```bash
# 1) Ingest the 134 new medium videos (16 seed already cached → skipped). ~3–5 GPU-h.
MBE_MANIFEST=eval/audiovisual/v2/video_manifest.json \
  scripts/run_ingest_4gpu.sh mbe-ingest 0,1,2,3 4

# 2) Teacher trajectories for the 120 new train videos (heldout NOT captured).
SHARD_DIR="$PWD/data/distillation/v2/teacher_shards" \
  scripts/run_teacher_4gpu.sh mbe-ingest 0,1,2,3 0.0

# 3) Combine reuse (49 medium pilot) + new v2 teacher trajectories.
cat data/distillation/raw_trajectories/full.trajectories.jsonl \
    data/distillation/v2/teacher_shards/traj_*.jsonl \
    > data/distillation/v2/combined_v2.trajectories.jsonl

# 4) Build SFT set; force the 30 heldout videos out of train; cap to C1.
conda run --no-capture-output -n mbe-ingest python scripts/distill_build_dataset.py \
  --trajectories data/distillation/v2/combined_v2.trajectories.jsonl \
  --include-tier-2 \
  --holdout-videos data/distillation/v2/heldout_cases.jsonl \
  --max-train-samples 1700 \
  --out-dir data/distillation/v2
# verify build_dataset_report.json: train_samples ∈ [1500,1700], val leak = 0.

# 5) Train (2 epochs). Checkpoint every 20 steps and KEEP ALL (--save-total-limit 0)
#    so the best can be picked by hard-set pass_rate, not just eval_loss. ~21 ckpts
#    × ~0.5-0.8GB ≈ 10-17GB. load_best_model_at_end still ships the min-eval_loss one.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n mbe-ingest python scripts/distill_train.py \
  --train data/distillation/v2/train.jsonl --val data/distillation/v2/val.jsonl \
  --epochs 2 --cutoff-len 8192 --eval-steps 20 --eval-cutoff-len 4096 \
  --save-total-limit 0 --wandb --run-name distill-v2-hardrebalance

# 6) Eval 3-way (base / pilot / v2) on the 90-case heldout, judge on, single vLLM.
#    cases = data/distillation/v2/heldout_cases.jsonl ; report pass_rate per modality.
```
Acceptance (spec §7): v2 ≥ pilot + 3pp (≈≥0.74); v2 `joint` ≥ pilot `joint`.

---

## WorldSense audio-visual enrichment (added 2026-06-03)

The medium-only Video-MME mix caps `joint` at ~23% (data-limited). To backfill the
audio-visual/temporal signal we pulled **WorldSense** (`honglyhly/WorldSense`,
on **ModelScope** + HF mirror; CC BY-NC 4.0): 1662 audio-visual-synchronised videos,
3172 MCQ, ships raw mp4 (18 GB) + subtitles + `worldsense_qa.json`.

**Modality mapping** (`scripts/build_worldsense_eval.py`, by `task_type`):
`audio` = Audio Source-Localization/Recognition/Counting/Change (409 q);
`joint` = temporal+emotion (Event Sorting, Temporal Localization/Prediction, Action
Counting, Object State Change, Causal Reasoning, Emotion Change, Human/Video Emotions,
Event Recognition — 1225 q); `visual` = the rest (1538 q). **~51% audio-or-temporal**
vs Video-MME's ~19%.

**Use (decided):** training enrichment + a *separate* audio-visual heldout (never
eval the orchestrator on data it trained on).

| set | videos | cases | mix (joint/audio/visual) |
|---|---|---|---|
| **train enrichment** (joint/audio-weighted) | 150 | 304 | 137/76/91 → **70% joint+audio** |
| **audio-visual heldout** (disjoint) | 40 | 91 | 32/45/14 → **85% joint+audio** |

**Artifacts:** `eval/audiovisual/worldsense/{questions.jsonl,video_manifest.json,
video_manifest.used.json}` (used = 190 ingest videos); `data/distillation/worldsense/
{train_cases.jsonl,heldout_cases.jsonl,teacher_shards/cases_{0..3}.jsonl}` (shards 73/77/78/76).
Download helper: `scripts/download_worldsense.py` (ModelScope-first). Raw zips kept in
`data/worldsense_src/` (deletable to reclaim 18 GB).

### Run book additions (GPU steps, run alongside the Video-MME v2 steps)
```bash
# 1b) Ingest the 190 used WorldSense videos (filtered manifest, not all 1662).
MBE_MANIFEST=eval/audiovisual/worldsense/video_manifest.used.json \
  scripts/run_ingest_4gpu.sh mbe-ingest 0,1,2,3 4

# 2b) Teacher trajectories for the 150 WorldSense train videos (heldout NOT captured).
SHARD_DIR="$PWD/data/distillation/worldsense/teacher_shards" \
  scripts/run_teacher_4gpu.sh mbe-ingest 0,1,2,3 0.0

# 3b) Fold WorldSense train trajectories into the combined set (with Video-MME v2).
cat data/distillation/raw_trajectories/full.trajectories.jsonl \
    data/distillation/v2/teacher_shards/traj_*.jsonl \
    data/distillation/worldsense/teacher_shards/traj_*.jsonl \
    > data/distillation/v2/combined_v2.trajectories.jsonl
# Then build (step 4 above) with both heldouts forced out:
#   --holdout-videos data/distillation/v2/heldout_cases.jsonl   (Video-MME)
# WorldSense heldout videos are never teacher-captured, so they can't leak; eval
# them separately. Projected train_samples jumps ~+2700 (304 cases × ~9) → likely
# OVER C1 — set --max-train-samples 1700 (or raise the cap deliberately).

# 6b) Extra eval slice: run base/pilot/v2 on the audio-visual heldout too.
#   cases = data/distillation/worldsense/heldout_cases.jsonl  (report per modality)
```
**Note on C1:** adding 304 WorldSense train cases (~150 videos) pushes projected
train_samples well past 1700. Either keep `--max-train-samples 1700` (drops whole
videos to stay in budget — mixes Video-MME + WorldSense proportionally) or raise the
cap as a deliberate scale decision; verify in `build_dataset_report.json`.

---

## Actual run results (2026-06-03 PM) — ingest → teacher → build done; train/eval paused

- **Ingest:** 340/340 cached on 4 GPUs. 5 videos initially failed because DashScope's
  content filter rejected a single scene frame with `data_inspection_failed`, which aborted
  the whole video. **Fixed** in `app/preprocess.py` (`_caption_one_tolerant`): a per-frame
  VLM rejection now drops that one caption + warns instead of failing ingest. Re-ingested
  the 6 affected (incl. **2 that were in the Video-MME heldout** — would have shrunk it).
- **Teacher:** v2 360/360 + WorldSense 304/304 captured (deepseek-v4-pro, AGENT v22),
  **~5.6 cases/min aggregate on 4 GPUs** (~43 s/case/shard), ~2 h wall.
- **Build:** 863 trajectories → **1499 train / 390 val samples**, 0 cap needed (in C1),
  **zero heldout leakage** (verified at source: 70 heldout videos never teacher-captured).

### Tier / quality analysis of the 863 trajectories
Tiering (`app/distill_filter.py`): tier_1 = correct+clean; tier_2 = correct but a runtime
guard fired (kept via `--include-tier-2`); tier_3 = **discarded** (teacher wrong, or cap/empty).

| source | tier_1 | tier_2 | tier_3 (dropped) | teacher correct% |
|---|---|---|---|---|
| reuse (pilot medium) | 22 | 14 | 17 | 67% |
| Video-MME (v2 new medium) | 247 | 99 | 160 | 68% |
| **WorldSense** | 103 | 32 | **169** | **44%** |
| **total** | **372** | **145** | **346 (40%)** | — |

Two findings that drive the methodology:
1. **All 346 dropped are `judge_wrong`** (teacher answered incorrectly) — none were cap/empty.
   So the training set is **100% teacher-correct**; teacher-*failed* data is fully excluded,
   which is correct for SFT imitation (imitating wrong answers teaches wrong reasoning).
2. **tier_2 is 100% `dedup`-only** (correct answer, one duplicated tool call deduped) — the
   most benign guard, no bad behavior taught. So `--include-tier-2` is well-justified here;
   dropping it would cut to ~1080 samples (below the C1 floor) for ~no quality gain.

**Teacher-capacity ceiling (key):** deepseek-v4-pro fails **56%** on WorldSense audio-visual,
so surviving WorldSense training data skews toward the *easier* joint cases the teacher could
solve. The hardest audio-visual signal is exactly what gets dropped. Per spec §7, the next
lever for hard A/V is a **stronger teacher / audio tool**, not more data or filtering.

---

## FINAL RESULT (2026-06-04) — 3-way eval complete

3-way on the 181-case heldout (single vLLM: base + pilot LoRA + v2 LoRA, judge on, 1 run).
W&B: training run `ts8urbus`, eval run `distill-v2-3way-eval` / `gi7iicvl`.

| pass_rate | base | pilot | v2 | v2−pilot |
|---|---|---|---|---|
| overall | 0.547 | 0.630 | **0.669** | **+3.9pp ✅** |
| joint | 0.362 | 0.621 | 0.621 | +0.0 |
| audio | 0.356 | 0.444 | 0.444 | +0.0 |
| visual | 0.785 | 0.708 | **0.815** | +10.8pp |
| overview | 0.846 | 0.923 | 0.923 | +0.0 |
| Video-MME | 0.700 | 0.722 | **0.800** | +7.8pp |
| WorldSense | 0.396 | 0.538 | 0.538 | +0.0 |

- **Primary acceptance MET**: v2 − pilot = +3.9pp (≥ +3pp). v2 is the best model.
- **Secondary (v2 joint ≥ pilot joint): equality, no gain.** v2's improvement over pilot is
  entirely `visual`/Video-MME; joint/audio/overview/WorldSense identical to pilot.
- **Confirms the teacher-capacity ceiling**: WorldSense A/V enrichment added no joint/audio
  over pilot because the teacher fails 56% on it. Next lever = stronger teacher / audio tool,
  not more data. Full analysis + resume bullets in docs/distill_outcomes_and_learnings.md.
