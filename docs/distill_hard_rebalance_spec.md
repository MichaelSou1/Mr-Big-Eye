# Distillation data spec — difficulty rebalance (hard videos), v2

**Status:** proposed (2026-06-03)
**Supersedes the approach in:** the short-video scale-up of 2026-06-02 (see [[distill-data-scaleup]]).
**Owner:** MichaelSou1

---

## 1. Why (what the last round taught us)

The 2026-06-02 scale-up grew the training set 4.5× (355 → 1618 samples) by adding
**197 short Video-MME videos** (cheap to ingest). On a **fair, leakage-free hard
eval** (16 unused *medium* videos / 48 questions, none in any model's training):

| model | train samples | hard-set pass_rate | vs base |
|---|---|---|---|
| base 7B | 0 | 0.562 | — |
| **pilot** (50 medium videos) | 355 | **0.708** | +14.6pp |
| **scaleup** (50 medium + 197 short) | 1618 | **0.667** | +10.4pp |

**The 4.5× scale-up did not help — it regressed −4.2pp vs the pilot on hard
cases.** On the dominant `visual` hard questions scaleup (23/37) fell back nearly
to base (22/37) while pilot (26/37) stayed ahead. Two causes:

1. **Difficulty/quality > quantity.** Short videos are near-trivial (base 7B
   already scores 0.833 on a short heldout). Their trajectories teach patterns the
   model already knows; they add tokens, not capability, and dilute the hard signal.
2. **Overfitting.** The scaleup adapter evaluated was epoch-3 (past the eval-loss
   minimum at epoch≈1.85). The best checkpoint was not retained.

Caveat: 48 cases / 2-case net delta is **not statistically strong** — read this as
"more data of the wrong kind didn't help," not "data hurts."

## 2. Objective & constraints

**Objective.** Improve hard-task pass_rate over the pilot (0.708) by rebalancing
the training mix toward **harder, more multimodal** videos — *without* growing
training cost.

**Hard constraints.**
- **C1 — training size ~constant.** Target **~1500–1700 train samples** (≈ current
  1618). Do not balloon sample count.
- **C2 — training wall-time bounded.** Medium/long trajectories are longer per
  sample, so equal sample count ≈ 1.3–1.6× tokens. Offset with **2 epochs** (the
  measured optimum anyway) and **`--cutoff-len 8192`**. Net wall-time ≈ the
  2026-06-02 run, not more.
- **C3 — a real held-out hard set** that *no* compared model trained on, for an
  apples-to-apples base / pilot / new comparison.

**Non-goals.** Re-running short videos; growing total data; changing the agent /
tool set / teacher model (stay **deepseek-v4-pro**, AGENT_CODE_VERSION v22 so old
trajectories remain combinable).

## 3. Target data composition

Available unused pool (verified 2026-06-03): **medium 235, long 300** videos free;
of unused medium+long, **330 videos contain ≥1 multimodal (`joint`) question**.

### 3.1 Training videos (~150 total, ~1600 samples)
Medium/long yield more agent steps per case → ~10–12 samples/video (vs ~6.5 for the
short-heavy mix), so **fewer videos hit the same sample budget**:

| source | videos | notes |
|---|---|---|
| **reuse** existing 50 medium (already ingested + trajectories) | 50 | pilot's strength; free |
| **add** new medium | ~70 | the workhorse difficulty |
| **add** new long (≤ ~30 min, cap the 1h ones) | ~30 | hardest; cap count — slow ASR/ingest |
| **total** | **~150** | → **~1500–1700 train samples** at 2-epoch budget |

**Modality stratification (the key change):** select videos to raise the
multimodal share. Target question-modality mix in the *kept* training set:
- `joint` (audio-visual / temporal): **≥ 35%** (was ~15%)
- `visual`: ~50%
- `overview`: ~15%

Prefer videos whose questions include `Temporal Reasoning / Action Reasoning /
Temporal Perception` (→ `joint`) and `Information Synopsis` (→ `overview`).

> **Drop short videos entirely** from training. If a short-regression check is
> wanted, keep them only in eval (§4), never in train.

### 3.2 Sample-budget control
The build naturally emits N samples; hit C1 by choosing the **video count** and, if
needed, the `distill_build_dataset.py --val-ratio` / a new `--max-train-samples`
cap (add if the build overshoots ~1700). Verify `train_samples` in
`build_dataset_report.json` before training.

## 4. Held-out hard eval (leakage-free)

- **~30 medium/long videos, ~90 questions**, modality-balanced, **never trained on**.
- **Seed it with the 16 medium videos already ingested + evaluated**
  (`data/distillation/hardeval_medium.jsonl`, 48 cases) so base=0.562 / pilot=0.708
  are reusable reference points; add ~14 more medium/long for ~90 cases.
- **Leakage control:** force these videos out of training with
  `distill_build_dataset.py --holdout-videos <heldout_cases.jsonl>` (added
  2026-06-03 — forces those video_ids into val, never train). Verify 0 leak as in
  the 2026-06-02 matched-split check.
- The held-out cases are **eval-only** (run by the three models); they need **no
  teacher capture**.

## 5. Pipeline (reuse existing scripts; deltas only)

All scripts honor `MBE_ROOT`; ingest/harness env = **`mbe-ingest`** (see
[[ingest-harness-env]]); cache lands in `/home/gpus/mbe_data/cache` (see
[[mbe-data-cache-location]]). 900 mp4s already downloaded in
`data/videomme_videos/` — **no re-download**.

1. **Sample** train + heldout videos:
   `scripts/sample_videomme.py --buckets medium,long --durations-file
   data/videomme_src/durations.json --target <N>` — extend it (or a thin wrapper)
   to also **stratify by modality** (over-weight `joint`). Long-video cap: exclude
   durations > ~1800s to bound ingest.
2. **Build** eval cases + manifest: `scripts/build_videomme_eval.py`.
3. **Ingest** (4-GPU): `scripts/run_ingest_4gpu.sh mbe-ingest 0,1,2,3 4`.
   ⚠️ medium/long are ~5–15 min/video; ~100 new × parallel/4 ≈ **3–5 GPU-h**.
4. **Teacher** trajectories (4-GPU): `scripts/run_teacher_4gpu.sh`. ~100 train
   videos × 3 ≈ 300 cases; medium/long → longer loops → more deepseek-v4-pro +
   DashScope-VLM calls/case. Held-out videos are **skipped** here.
5. **Combine** new trajectories with reusable old ones (the 50 medium) →
   `distill_build_dataset.py --trajectories <combined> --include-tier-2
   --holdout-videos <heldout> --out-dir data/distillation/v2`.
6. **Train**: `scripts/distill_train.py ... --epochs 2 --cutoff-len 8192
   --eval-steps 20 --eval-cutoff-len 4096 --wandb` (eval auto-keeps best ckpt via
   `load_best_model_at_end`, added 2026-06-03).
7. **Evaluate**: serve base + pilot + v2 on one vLLM
   (`--lora-modules pilot=ckpt_full v2=<new>`, `--enforce-eager
   --gpu-memory-utilization 0.90`), run `eval_harness.py` 3× on the hard heldout,
   compare.

## 6. Training config (frozen knobs)

QLoRA r32/α64, lr 1e-4 cosine, batch 1 × grad-accum 8, bf16, liger fused CE,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **2 epochs** (epoch-3 overfits).
`--eval-steps 20 --eval-cutoff-len 4096` for a W&B held-out-loss trend; rely on
`load_best_model_at_end` to ship the lowest-eval_loss checkpoint.

## 7. Evaluation protocol & acceptance

- 3-way (base / pilot / v2) on the **same ~90-case hard heldout**, judge on, single
  vLLM, identical `.env` for VLM/judge (only orchestrator differs).
- Report pass_rate overall **and per modality** (the regression last time was
  modality-specific).

**Acceptance (primary):** `v2 ≥ pilot + 3pp` on the hard heldout
(i.e. ≥ ~0.74), with the W&B `eval/loss` curve showing a controlled minimum (no
late-epoch divergence shipped).
**Secondary:** v2's `joint` (multimodal) pass_rate ≥ pilot's — the rebalance should
help most where audio-visual reasoning matters.
**Negative result is acceptable and informative:** if v2 still ≤ pilot, the
bottleneck is not data difficulty but the 7B student / LoRA capacity or the teacher,
and the next lever is model/teacher, not data.

## 8. Cost & risks

- **Ingest:** ~3–5 GPU-h (medium/long slow ASR). Disk: 900 mp4s already local;
  cache grows ~modestly. 228 GB free as of 2026-06-03.
- **Teacher API:** ~300 train cases × longer loops → more deepseek-v4-pro spend than
  the 591-short-case round per case; budget accordingly. Confirm before launch.
- **Risk — long-video context blowups:** 1h videos → huge transcripts/trajectories
  may exceed cutoff and get truncated/dropped at build. Mitigation: cap duration
  ≤ ~30 min, watch `encoded train=x/y dropped overlong` at train start.
- **Risk — sample budget drift:** medium/long emit more samples/video; if
  `build_dataset_report.json:train_samples` ≫ 1700, drop videos or add a
  `--max-train-samples` cap to honor C1.
- **Risk — small-N noise:** ~90 heldout cases tightens but doesn't eliminate the
  variance seen at 48; treat <3pp deltas as ties.

## 9. Open knobs (decide at build)

- medium:long split (default ~70:30; more long = harder + slower).
- exact `joint` target (default ≥35%).
- whether to keep a short-video eval slice for an explicit easy/hard contrast.
- final train video count to land `train_samples` in [1500, 1700].
