# N3 PREREGISTRATION — natural-image (V*-style) pole probe

> **Committed BEFORE the probe is run** (2026-06-25). Predictions, selection
> criteria, and the decision rule are fixed here; results go in a separate
> `n3_complete_0625.md`. This honors the chapter_plan red-line "先写预注册 verdict
> 再看结果" and the prereg-two-directions requirement.

## The one question N3 must answer
WU-3 §4.2 found that **masking the chart** ("depriving re-reading" artificially)
makes the model **hallucinate** (8B `other` .02→.19, 9×) rather than fall back on
its written CoT (`follow` only .01→.09). So the *masked* condition is an **oracle**
for "what happens when re-reading is removed."

The open question — and N3's only new contribution — is:

> **Is masked-chart's "deprive-reread → hallucinate" an artifact of the artificial
> `mask-image` operation, or does it reproduce on _naturally_ non-rereadable
> images (small targets in natural scenes)?** And when re-reading is naturally
> hard, does the model instead lean on its text CoT (`follow↑` = a clean
> *load-bearing* pole, the 2510.23482 mirror), or does it still just fabricate
> (`other↑`)?

This turns the planned clean **bipolar** map (charts: CoT decorative / V*: CoT
load-bearing) into a sharper, more honest **tristate** organizing finding (see
"Regime axis" below) — regardless of which way the result falls.

## Operationalization (fixed)
- **Substrate**: TallyQA **complex** (`is_simple=False`) counting on Visual-Genome
  natural images — "how many people are wearing blue shirts?" = perceive + filter +
  count, the natural-image analog of charts' perceive + arithmetic. Free-form
  **integer** answers → identical relaxed-numeric grader and the SAME
  corrupt-number / snap-follow-other probe as ChartQA & TabMWP; **no MCQ → no
  letter-luck / disguised-accuracy** (2402.14897).
- **Why not real V*Bench**: V*Bench is multiple-choice (attribute / spatial),
  incompatible with the numeric corrupt probe and the snap/follow/other split.
  TallyQA-complex keeps the *scientific* property (small targets in natural scenes,
  hard to re-perceive) while keeping the probe + grader identical across all three
  regimes.
- **Selection** (`scripts/build_natcount_test.py`): complex, integer answer
  ∈ [2,20], ≤2 cases/image, scanning test shards in order → **n=400**.
  Realized: answer dist {2:228, 3:77, 4:46, 5:20, 6:8, 7:4, 8:5, 9:3, 10:3, …},
  388 unique images, all AMT human-written complex Qs, grader self-check 400/400,
  mcq=0, letter-luck=0.
- **Model = GENERAL BASE** (Qwen3-VL-8B-Instruct, then -32B-Instruct), **no
  chart-SFT adapter** (`battery_n400.py --adapter none`). A chart-SFT student is
  out-of-domain on natural images → near-all-wrong → empty kept set; the
  rereadability question is about the base perception policy anyway.
- **Probe** (identical to WU-2/WU-3): for cases the base gets right *with* a CoT,
  corrupt one intermediate number / shuffle the CoT, force-continue, classify the
  post-corrupt answer **snap** (==gold, re-perceived) / **follow** (==injected,
  CoT load-bearing) / **other** (neither, fabricated). Run **present + masked**.
  Controls: shuffle, filler, truncate, delete (paraphrase skipped — needs the
  DeepSeek API; not load-bearing for the follow/other claim).
- **Known limitation** (recorded now): VG images are median ~500px, not the 4K
  tiny-0.1%-target of real V*. So this is the *reasoning-counting* natural-image
  regime, not the most extreme resolution-bottleneck. Scope-gate: corroborate the
  bi/tri-pole, not a full natural-image study.

## Preregistered predictions (BOTH directions reported)
Reference (existing, image-present): chart/tabmwp 8B present **snap .82–.97,
follow ≤.01, other ≤.02** (re-read bypass); masked → **other↑** (.19 / 8B ChartQA).

- **H_pole** (clean positive pole = 2510.23482 mirror): on natcount **present**,
  `follow↑` and `other` stays low, `snap` drops well below the chart .82–.97 —
  i.e. the model, unable to cheaply re-perceive small natural targets, **relies on
  its text CoT**. This would give a clean third regime (load-bearing).
- **H_collapse** (masked-oracle prior — judged **more likely**): on natcount
  **present**, `other↑` (fabrication), `follow` NOT high — i.e. naturally-hard
  perception behaves like a masked chart; there is **no clean load-bearing pole**,
  the honest map is bypass/fabricate with load-bearing (if any) only in a narrow
  band.
- Secondary (both H): `snap` on natcount present < chart present (natural targets
  are harder to re-read); masked natcount `other` ≥ present natcount `other`.

## Regime axis this populates (rereadability → 3 states)
| state | signature | exemplar |
|---|---|---|
| **bypass** (re-read) | snap high, follow~0, other~0 | charts/tables present (CoT decorative) |
| **fabricate** (hallucinate) | other high | charts masked (re-read removed artificially) |
| **load-bearing** (text CoT carries) | follow high, other low | predicted V*/natural pole — **N3 tests if this state exists** |

N3's verdict = **where natcount-present lands** on this axis.

## Acceptance
- [ ] natcount **base** free-form acc + n_eval (present), 8B (and 32B if feasible).
- [ ] snap/follow/other for natcount **present + masked**, reported next to
      chart/tabmwp cells (`scripts/battery_followrate.py`).
- [ ] Verdict written for whichever of H_pole / H_collapse holds; the tristate
      regime axis filled in with natcount's position — **either outcome is a
      result** (clean pole, or "fabricate reproduces on natural images → no clean
      pole, regime is tristate with load-bearing only in a narrow band").
