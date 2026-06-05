# Distillation v2 — 3-way evaluation report

**Date:** 2026-06-04/05 · **Models:** base (Qwen2.5-7B-Instruct) / pilot / **v2** ·
**W&B:** training `ts8urbus`, eval `gi7iicvl`.

## Protocol
- **Heldout:** 181 cases, **leakage-free** — 90 Video-MME + 91 WorldSense; no heldout video
  was ever teacher-captured or trained on (verified 0 leak at the trajectory source).
- **Apples-to-apples serving:** one vLLM hosting the base model + `pilot` and `v2` LoRA
  modules (hot-swap); identical VLM + judge endpoints; only the orchestrator differs.
- **Metric:** LLM-judged answer pass_rate, reported overall and per **modality**
  (`joint` = audio-visual+temporal, `audio`, `visual`, `overview`) and per **source**.
- 1 run per model; per-modality n is 13–65 → treat **<3pp deltas as ties**.

## Overall

| model | overall pass_rate | vs base |
|---|---|---|
| base (no distill) | 0.547 (99/181) | — |
| pilot (355-sample) | 0.630 (114/181) | +8.3pp |
| **v2 (this round)** | **0.669 (121/181)** | **+12.2pp** |

**Acceptance:** primary `v2 ≥ pilot + 3pp` **MET (+3.9pp)**. v2 is the best model.

## By modality

| modality (n) | base | pilot | v2 | v2−base | v2−pilot |
|---|---|---|---|---|---|
| joint (58) | 0.362 | 0.621 | 0.621 | **+25.9pp** | +0.0 |
| audio (45) | 0.356 | 0.444 | 0.444 | +8.9pp | +0.0 |
| visual (65) | 0.785 | 0.708 | 0.815 | +3.1pp | **+10.8pp** |
| overview (13) | 0.846 | 0.923 | 0.923 | +7.7pp | +0.0 |

## By source

| source (n) | base | pilot | v2 |
|---|---|---|---|
| Video-MME (90) | 0.700 | 0.722 | **0.800** |
| WorldSense (91) | 0.396 | 0.538 | 0.538 |

## Findings
1. **Distillation closes most of the base 7B's audio-visual/temporal gap:** `joint`
   0.362 → 0.621 (+25.9pp), `audio` +8.9pp. The agentic tool-use policy transfers.
2. **v2 is best overall (+3.9pp over pilot), but its gain over pilot is entirely `visual`/
   Video-MME;** on joint/audio/overview/WorldSense it equals pilot (same cases passed). The
   WorldSense audio-visual enrichment added **no joint/audio over pilot**.
3. **Teacher-capacity ceiling:** the teacher (deepseek-v4-pro) fails ~56% on WorldSense, so
   the student cannot exceed it there — the audio-visual bottleneck is the teacher/tooling,
   not data volume. Next lever = stronger teacher / a non-speech audio tool.

## Checkpoint selection & the epoch-3 question (2026-06-05)
All 20-step checkpoints were retained. To test whether v2 (epoch-2, the eval-loss minimum
`ckpt-320`) was under-trained — eval loss was still falling at the end — we ran a cheap
**pass_rate trajectory** on a 60-case stratified subset:

| ckpt | epoch | subset pass_rate |
|---|---|---|
| ckpt-160 | 1.0 | 0.717 |
| ckpt-240 | 1.5 | **0.733** (subset peak) |
| ckpt-280 | 1.7 | 0.667 |
| ckpt-320 (shipped) | 2.0 | 0.683 |

pass_rate **peaks ~epoch 1.5 then plateaus/declines** while eval loss keeps falling — a
direct demonstration of "eval-loss min ≠ task max." So **epoch 3 was rejected** (would
overfit). The subset peak at ckpt-240 was then checked on the **full 181**: ckpt-240 =
**0.657** vs ckpt-320 **0.669** (Δ −1.1pp, within noise) — the subset 0.733 was small-N
noise. **Shipped checkpoint stands: ckpt-320 (epoch 2, eval_loss 0.4044).**

This two-step probe (≈2.5 GPU-h) avoided both a wasted ~11 GPU-h epoch-3 run and a
noise-driven wrong re-ship — the payoff of "keep all checkpoints, select by pass_rate."
