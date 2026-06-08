# Mr. Big-Eye distillation — evaluation summary (as of 2026-06-03)

Consolidated task-level (agent pass-rate) results for the orchestrator distillation
effort. All runs use the same harness (`scripts/eval_harness.py`,
AGENT_CODE_VERSION v22), the same VLM tool backend (DashScope `qwen3-vl-plus`) and
LLM judge (doubao-seed-2-0-lite); **only the orchestrator model differs** between
rows. Pass-rate = LLM-as-judge correctness over the held-out cases.

Models compared:
- **base 7B** — plain `Qwen2.5-7B-Instruct`, no adapter.
- **teacher** — `deepseek-v4-pro` as orchestrator (the distillation target / ceiling).
- **pilot** — student LoRA `ckpt_full`, trained on **355 samples** (50 medium videos).
- **scaleup** — student LoRA `ckpt_scaleup`, trained on **1618 samples** (50 medium + 197 short videos), epoch-3.

---

## 1. Headline

> **More data of the wrong kind did not help.** Quadrupling the training set with
> cheap *short* videos (355→1618 samples) produced a model that **regressed −4.2pp
> vs the pilot** on a fair hard eval. Difficulty/quality of the training videos
> matters more than raw sample count.

The pilot (`ckpt_full`) remains the best task model to date.

---

## 2. Hard held-out eval — the decisive comparison (2026-06-03)

16 **medium** Video-MME videos / **48 questions**, none seen in *any* model's
training (leakage-free). This set discriminates (base ≠ ceiling).

| model | train samples | pass_rate | vs base |
|---|---:|---:|---:|
| base 7B | 0 | **0.562** | — |
| **pilot** | 355 | **0.708** | **+14.6pp** |
| scaleup | 1618 | 0.667 | +10.4pp |

**scaleup − pilot = −4.2pp** (net 2 cases of 48). Per-modality:

| modality | n | base | pilot | scaleup |
|---|---:|---:|---:|---:|
| visual | 37 | 22 | **26** | 23 |
| joint (multimodal) | 5 | 1 | 3 | **5** |
| overview | 6 | 4 | **5** | 4 |

On the dominant `visual` hard questions scaleup (23/37) fell back near base
(22/37); pilot (26/37) stayed clearly ahead. scaleup led only on `joint` (5/5, but
n=5). Files: `data/distillation/runs/{base7b,pilot,scaleup}_hard48.json`.

> Caveat: 48 cases / 2-case delta is **not statistically strong** — read as
> "scaleup is not better," not "significantly worse."

---

## 3. Easy held-out eval — inconclusive (ceiling) (2026-06-03)

48 **short** videos / 48 questions (scaleup's own held-out split; mostly `visual`).

| model | pass_rate |
|---|---:|
| base 7B | 0.833 |
| scaleup | 0.854 |

Both near ceiling (+2.1pp = 1 net case); short videos are too easy to discriminate
(base 7B alone already 0.833). This is *why* §2 used medium videos. Files:
`{base7b,scaleup}_sub48.json`.

---

## 4. Original held-out eval — pilot vs teacher vs base (2026-05-29)

36 questions (medium videos), the original Phase D held-out set.

| model | pass_rate | note |
|---|---:|---|
| **teacher** (deepseek-v4-pro) | **0.833** | distillation ceiling |
| **pilot** (student, 355 samples) | **0.722** | recovers most of base→teacher gap |
| base 7B | 0.611 | floor |

The pilot closed **~50%** of the base→teacher gap (0.611 → 0.722, ceiling 0.833) —
distillation works. pilot's 0.722 here and 0.708 on the fresh hard set (§2) are
consistent, so the hard-set scale is trustworthy. Files:
`{baseline,student_full,base7b}_heldout.json`.

> Not directly comparable to §2/§3 (different case set), but the pilot's stability
> across two medium sets (0.722 / 0.708) anchors the numbers.

---

## 5. Training run health (scaleup, W&B `qwen2.5-7b-distill-scaleup`)

- 1618 train / 417 val samples; 552 steps (3 epochs); QLoRA r32/α64, lr 1e-4, liger.
- **train/loss** 1.0 → ~0.22 (steady).
- **eval/loss** (new this round, OOM-safe cutoff 4096) bottomed ~0.42 at **step
  ~340 (epoch≈1.85)** then rose to ~0.45 by epoch 3 → **mild overfitting; epoch 2
  is optimal.** The evaluated `ckpt_scaleup` is the overfit epoch-3 model (best
  checkpoint was not retained at the time — since fixed via
  `load_best_model_at_end`).

The pilot (2026-05-29, `qwen2.5-7b-distill-pilot`) trained with `--no-eval`, so it
had no overfit visibility — a gap this round closed.

---

## 6. Smoke runs (not meaningful — n=2)

`student_smoke` (1.0) and `student_smoke_test` (0.5) are 2-case pipeline smoke
tests; ignore for quality conclusions.

---

## 7. Conclusions & next step

1. **Distillation works**: both students beat base 7B on hard cases (+10–15pp); the
   pilot recovers ~half the base→teacher gap.
2. **The short-video scale-up failed** its goal: 4.5× data, −4.2pp vs pilot on hard
   cases. **Quality/difficulty > quantity.**
3. **Overfitting** was a secondary factor (epoch-3 model shipped); now detectable and
   auto-corrected (`load_best_model_at_end`, `--eval-steps`).
4. **Next plan:** [`docs/distill_hard_rebalance_spec.md`](../docs/distill_hard_rebalance_spec.md)
   — keep ~1600 train samples but swap short→medium/long + multimodal, hold out a
   ~90-case hard set, train 2 epochs. Acceptance: v2 ≥ pilot + 3pp on the hard set.
5. **Best current model:** pilot `data/distillation/ckpt_full`.

## Appendix — raw run index

| file | set | n | date | pass |
|---|---|---:|---|---:|
| base7b_hard48.json | hard medium | 48 | 06-03 | 0.562 |
| pilot_hard48.json | hard medium | 48 | 06-03 | 0.708 |
| scaleup_hard48.json | hard medium | 48 | 06-03 | 0.667 |
| base7b_sub48.json | easy short | 48 | 06-03 | 0.833 |
| scaleup_sub48.json | easy short | 48 | 06-03 | 0.854 |
| baseline_heldout.json (teacher) | orig medium | 36 | 05-29 | 0.833 |
| student_full_heldout.json (pilot) | orig medium | 36 | 05-29 | 0.722 |
| base7b_heldout.json | orig medium | 36 | 05-29 | 0.611 |

All under `data/distillation/runs/`.
