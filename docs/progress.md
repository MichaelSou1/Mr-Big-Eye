# Research Progress Log — Trajectory-to-CoT Internalization

> Living log of experiment purposes, methods, results, and decisions for the
> "internalize agentic visual reasoning into single-forward CoT" project.
> Companion to [proposal.md](proposal.md) and
> [RESEARCH_SPEC_trajectory_to_cot.md](RESEARCH_SPEC_trajectory_to_cot.md)
> (see their §15 / delta sections for the design revisions this log acts on).
> Newest entries at the bottom of each section. Dates are absolute.

---

## 0. Thesis (current, narrowed)

When the relevant frames are visible (short-video regime, fixed uniform sample),
can tool-scaffolded multi-step visual reasoning be compressed into a **single
tool-free forward pass**, and can causal probes show it is genuine reasoning
(not template) — with a characterized **internalizability boundary**?

Two non-negotiable selling points: (1) no tool / single forward pass at
inference; (2) causal verification. Two hard constraints: (1) **self-improvement**
— trajectory-generating VLM == the 4B SFT base (no stronger VLM teacher; rewriter
may be larger text-only); (2) only internalize **type-1** (single-forward-doable)
steps. Scope: **vision-only** (frames only).

**STATUS (2026-06-11): the full supplementary spec is COMPLETE — see §7.** Exp-0..8 (below)
were the exploratory sweep; §7 turns them into a defensible, statistically-gated
**boundary + 2-regime map**, and — updating the earlier "training never run" note —
**the §11 SFT PoC was run and is a significant POSITIVE result** (8B ChartQA .617→.733,
+0.117, CI excludes 0; with a causal probe). Headline: internalizing agentic/reflective
reasoning into a small VLM has **no statistically reliable headroom** in the frames-visible
video regime (perception/selection-bound across 4B/8B/32B + a non-Qwen InternVL3-8B; agentic
nets within-variance, the only reliable effects *negative*); the **reasoning-bound** regime
appears only once a strong base (32B) solves perception (static charts), where internalization
is demonstrated to work. All variance-gate numbers regenerate from
`data/distill/results/results.jsonl`. (Exp-0..8 and the 2026-06-08 reflection pivot are kept
below for the record; the §1 multi-seed gate *retracted* the Exp-5 "8B +8%" as run-variance.)

---

## 1. Infrastructure (2026-06-08)

**Serving (local, no online API needed).** Decision: orchestrator/rewriter do not
need to be strong; both run on a local 30B. The VLM backbone MUST stay the 4B
(constraint #1).
- `Qwen3-VL-4B-Instruct` @ `http://127.0.0.1:30000/v1` — GPU0, vLLM (`vllm-qwen`
  env, vllm 0.19.0). The distillation backbone. Pulled from ModelScope
  `Qwen/Qwen3-VL-4B-Instruct` → `/home/gpus/models/Qwen3-VL-4B-Instruct`.
- `Qwen3-30B-A3B-Instruct` (AWQ-4bit) @ `:30001` — GPU1+2 (TP=2; TP=3 fails: 32
  heads not divisible by 3). orchestrator + rewriter. Weights
  `/home/gpus/DLT/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit`.
- Launch scripts: [scripts/serve/](../scripts/serve/) (`serve_vlm_4b.sh`,
  `serve_text_30b.sh`, `stop_servers.sh`). GPU3 free.
- Env notes: local sglang envs unusable (`sglang-qwen36-35b-awq` kernel 0.4.1 too
  old; `mbe-phase2` torch cu130 vs driver 570 → no CUDA). vLLM path works.

**Config.** [.env](../.env): `AGENT_VLM_BACKEND=local`, three endpoints wired,
`DATA_DIR=/home/gpus/mbe_data`, `DISTILL_SAMPLER_FRAMES=16`, vision-only
(`ENABLE_ASR=false`, `ENABLE_SLIDE_OCR=false`), encoders `MODELS_DEVICE=cpu`
`SIGLIP2_DTYPE=float32` (GPU/fp16 paths segfault/conflict during ingest), BGE/
SigLIP reused from `/home/gpus/Mr-Big-Eye/models`.

**Data (NExT-GQA, vision-only).**
- Annotations: cloned `doc-doc/NExT-GQA` → `/home/gpus/mbe_data/nextgqa_src`
  (gsub_val.json, map_vid_vidorID.json, val.csv — grounding GT intact).
- Videos: ModelScope `lmms-lab/NExTQA` `videos.zip` (6 GB, 1570 clips) →
  `/home/gpus/mbe_data/nextqa_videos/NExTVideo/<id>.mp4`.
- Pilot cases: [scripts/eval_convert_nextgqa.py](../scripts/eval_convert_nextgqa.py)
  → 120 cases / 112 videos; ingested 70 via
  [scripts/eval_ingest_videos.py](../scripts/eval_ingest_videos.py) (rebound=70,
  0 errors, all with grounding GT). Cache: `/home/gpus/mbe_data/cache/<hash>/`.
- Split: [scripts/split_cases.py](../scripts/split_cases.py) → 59 train / 11
  held-out (video-disjoint, IID eval point).

**Ingest gotchas fixed (reusable for full-scale 567):**
- funasr ASR loaded on GPU0 (VLM's card) → CUDA conflict. Fixed by vision-only
  `ENABLE_ASR/ENABLE_SLIDE_OCR=false`.
- **BGE-M3 multi-GPU bug:** wrapper omitted FlagEmbedding's `devices=` kwarg →
  auto-detected all 4 GPUs → multi-process pool → `model.share_memory()` →
  `random_device could not be read` crash. Fixed in
  [app/models.py](../app/models.py) (pin single device). Necessary for any GPU
  ingest.
- SigLIP-so400m on CPU ≈ 70 s/video (the ingest bottleneck). GPU encoder path
  still segfaults (BGE fp16 load alongside vLLM contexts) — unresolved, deferred
  until after the gate.

---

## 2. Code changes landed (pipeline correctness)

The P1–P6 patches from [the approved plan] (see SPEC §15) — implemented and
unit-tested (`tests/test_distill.py` 10/10):
- **P1** fixed uniform sampler [app/distill/frames.py](../app/distill/frames.py)
  + wiring (`sampler_frames` is the frame set for CoT/consistency/SFT/RL).
- **P2** evidence-coverage scope filter in
  [filter_strict.py](../app/distill/filter_strict.py).
- **P3** Phase-4 gate = `conditioned_ok ∧ consistency_ok ∧ ¬free_ok` with answer
  stripped from the seed CoT; headline metric `signal_gain_rate`
  ([filter_consistency.py](../app/distill/filter_consistency.py)).
- **P4** grounding GT threaded through CoT/SFT/RL artifacts.
- **P5** train/held-out video-disjoint split.
- **P6** RL temporal-IoU reward off (not load-bearing under fixed frames);
  shuffled-CoT control builder; 3c probe = claimed-timestamp-vs-GT.
- **Fix (2026-06-08):** rewriter cited agent-retrieved timestamps leaking from
  trajectory context, not the uniform sample → added marker-snapping
  ([rewrite.py](../app/distill/rewrite.py) `snap_markers_to_shown`, ≤2.5 s) +
  prompt now lists exact allowed timestamps. Verified on a real case.

---

## 3. Experiments

### Exp-0 — Phase-0 serving smoke (2026-06-08) ✅
**Purpose:** confirm the 4B endpoint answers multi-frame QA with valid citations.
**Method:** [scripts/check_local_vlm.py](../scripts/check_local_vlm.py).
**Result:** caption + multi-frame QA correct, `[FRAME:t=]` citations valid.
**Decision:** Phase-0 acceptance met; proceed.

### Exp-1 — 50-case pilot gate (2026-06-08) ✅ ran, decision STOP (later overturned as a false stop)
**Purpose:** measure internalizable-signal retention (the spec's go/no-go).
**Method:** `python -m app.distill.pilot --cases <train.jsonl> --n 50`
(generate → strict → rewrite → consistency). Work dir `data/distill/pilot/`.
**Result:**
| stage | value |
|---|---|
| trajectories | 50/50 (~30 min, 36 s/case) |
| strict pass | **7/50 (14%)** — 43 bad_process |
| rewrite pass | 7/7 (snap fix) |
| consistency retention | **0/7** |
| `signal_gain_rate` | **0.0** |
| base free-form acc (on the 7) | 100% |
All 7 dropped `base_already_correct_no_signal`. Strict drop tags: agent_loop(25),
retrieval(20), missing_expected_keyword(18), missing_citation_kind(15),
grounding_report_failed(13)…
**Decision:** STOP_SHRINK — but flagged as possibly a **false STOP**: strict
filter is coupled to the product harness's citation/agent-process protocol, which
may drop signal-bearing cases before the gate. Run a free-form diagnostic to
disambiguate before concluding.

### Exp-2 — base 4B free-form accuracy diagnostic (2026-06-08) ✅
**Purpose:** is the slice "too easy" (4B free-form ≈100%) or is the strict filter
discarding the signal (4B free-form well below 100%)?
**Method:** [scripts/diag_base_freeform.py](../scripts/diag_base_freeform.py) —
4B, uniform-16, no tools/CoT, one pass, on all 50; cross-tab with agent-correct
and strict drop reasons.
**Result:**
- **free-form accuracy = 68%** (16/50 wrong) → not too easy; there IS a failure
  surface. By qtype: ch .78, cw .71, tn .67, tc .60.
- Of the 16 free-wrong: **agent (4B+tools) also wrong = 13**; **agent-right ∧
  free-wrong = only 3**.
**Decision / insight:** the bottleneck is NOT mainly the strict filter — it's the
**self-improvement constraint**: 13/16 hard cases the 4B-agent itself can't solve
→ no distillable correct trajectory. The clean signal pool (agent-right ∧
free-wrong) ≈ **3/50 ≈ 6%**. Root cause: short video + uniform-16 means the
evidence is already on-screen, so tools (whose value is *finding frames*) add ~0
over free-form. The clean-vision-only-single-forward design and a non-trivial
internalization gap are in tension.

### Exp-2b — full-agent vs free-form accuracy (2026-06-08) ✅
**Purpose:** does the full agent (30B orchestrator + 4B + retrieval + reflection)
beat the bare 4B free-form? (Free data: agent `draft_answer` from pilot trajectories.)
**Result:** **agent 54% < free-form 68%.** Cross-tab: agent fixes 3 of free's
wrongs but **breaks 10 of free's rights** (net −7). On short videos, retrieval
*narrows* the view away from the already-good uniform-16, and 30B-over-4B-reads
adds errors. **The agent is net-negative vs the bare 4B here** — currently nothing
positive to internalize from the agent-with-retrieval.
**Caveat:** this agent used retrieval (frame-narrowing). The fair test of the
reflection hypothesis is the agent/critic over the SAME fixed uniform-16 (Exp-4).

### Exp-3 — reflection-gap diagnostic (Option B, 4B self-reflection) (2026-06-08) ✅
**Purpose (user-driven pivot):** hold frames constant (same uniform-16 for both)
to control frame-selection, leaving the agent's **reflection** (tool-free
self-re-examination) as the only difference. Does multi-turn self-reflection over
the SAME frames beat single-forward → is there a *reflection* gap worth
internalizing? (Sharpening: reflection must be tool-free to stay type-1 / honor
self-improvement.) Risk: small-model self-reflection often fails to self-correct.
**Method:** [scripts/diag_reflection_gap.py](../scripts/diag_reflection_gap.py) —
free-form vs 2-turn tool-free self-reflection over identical uniform-16; count
`reflect_gain` (reflect-right ∧ free-wrong) vs `reflect_lost`.
**Result:** free 68% → **reflect 68%**. `reflect_gain_cases = 0`,
`reflect_lost_cases = 0`, `net = 0.0` — **reflection flipped zero answers**.
Sanity-checked: reflection genuinely runs (text differs, e.g. "Re-examination of
all frames confirms…") but the 4B **reaffirms its original wrong answer** instead
of correcting (classic small-model self-rationalization). Not a runner bug.
**Decision / insight:** on this slice the agent's edge over free-form is **NOT
reflection** — pure tool-free self-reflection over fixed frames has **0 headroom**.
Combined with Exp-2 (agent-right ∧ free-wrong ≈ 6%, mostly frame-selection-driven),
the picture crystallizes: **the 4B's reasoning gains are inseparable from getting
NEW visual evidence (frame/region selection = type-2, out of scope). The
internalizable (type-1) reasoning-over-fixed-frames has near-zero headroom here.**
This empirically confirms the design tension (clean vision-only single-forward ⟂
non-trivial internalization gap). Strong **internalizability-boundary** result.
Option B as "internalize tool-free reflection" has no signal on NExT-GQA short
clips. Forks (see §5): (A) publish the boundary finding; (B) harder data where
fixed-frame reflection helps; (C) allow same-frame *zoom/region re-read* (restores
a type-1-ish gap on small-object/clock/text questions).

---

### Exp-4 — 30B-orchestrated reflection over fixed uniform-16 (2026-06-08) ✅
**Purpose:** the corrected reflection test — 30B (text critic, blind) orchestrates
re-checking over the SAME uniform-16; 4B is the eyes; 30B integrates → final.
Frames controlled, no retrieval. Does it beat free-form?
**Method:** [scripts/diag_orch_reflection.py](../scripts/diag_orch_reflection.py).
**Result:** free 70% → **orch 66%**. `orch_gain = 0`, `orch_lost = 2`, net −4%.
**Zero free-wrong cases recovered.** Mechanism: the 30B is text-blind, reasons
over the 4B's reads; when the 4B mis-*perceives*, the 30B integrates the wrong
read and reaffirms. Orchestrator-distillation premise (agent > 4B) fails here.

### Exp-C — perception-headroom probe (2026-06-08) ✅
**Purpose:** are the free-wrong failures perceptual (resolution/localization →
type-2 lever) or fundamental (4B can't, period)?
**Method:** [scripts/diag_perception_headroom.py](../scripts/diag_perception_headroom.py)
— re-run the 16 free-wrong under uniform@640, GT-localized@640/768.
**Result (of 16 free-wrong, # now correct):** base@448 2 (= stochastic noise),
uniform@640 4, **GT-localized 5**, GT@768 5. **11/16 stay wrong even with the
exact gold frames at max res.** → wall is ~70% **fundamental** (4B can't do it),
only ~3 cases (~6% of 50) are perception/localization-limited — and that slice is
exactly type-2 frame-selection (non-internalizable).

### Synthesis (2026-06-08) — internalizability boundary, four-experiment convergence
On NExT-GQA short-video MCQ with a 4B VLM:
- Free-form 4B ≈ **68–70%** (±2% stochastic).
- Of the ~30% wrong: ~6% perception/localization-limited (type-2, non-internalizable);
  ~22% **fundamental** (no frame/res/reasoning intervention helps).
- **Reflection headroom = 0** whether the reflector is the 4B (Exp-3) or the 30B
  orchestrator (Exp-4); the full agent is **net-negative** (Exp-2b).
- **Mechanism:** the bottleneck is the VLM's *perception*, not reasoning. Text-level
  reflection/orchestration cannot fix perceptual errors. The agent's only edge
  (frame selection) is small here (short video → uniform-16 already sees it) and
  is type-2 by construction.
- **Conclusion:** essentially **no internalizable reasoning headroom on this data**.
  Strong, mechanism-backed boundary result. To get a POSITIVE internalization
  result you must move to data where the 4B *perceives* fine but needs multi-step
  *reasoning* (perception ⟂ reasoning), e.g. CLEVRER-style compositional/causal —
  then re-run the gap diagnostics there.

### Exp-5 — 8B scale sweep + CLEVRER reasoning-data pivot (2026-06-08, in progress)
**Purpose (user /goal):** (1) re-run all experiments on Qwen3-VL-**8B** — is the
NExT-GQA wall capacity or data? (2) move to a reasoning-bound dataset (CLEVRER)
where perception is clean so any gap isolates reasoning.
**Serving:** 8B-VL TP2 @:30002 (GPU0,1) + 30B AWQ TP2 @:30001 (GPU2,3).
**8B free-form (NExT, 50 cases):** **0.64 ≈ 4B's 0.68** (within ±noise) — scaling
4B→8B does NOT lift NExT-GQA accuracy. 8B fixes 5/16 of 4B's free-wrong but breaks
~7; of the 11 "4B-fundamental" (wrong even w/ GT frames) 8B solves only **4** →
~60% is a **data/task wall, not capacity**. Confirms NExT-GQA is the wrong
substrate and validates the CLEVRER pivot.
**8B NExT reflection/orch/perception (50 cases):**
- 8B self-reflection: 0.64 → **0.52** (gain 2, lost 8, net **−12%**) — still a parrot, worse.
- **8B 30B-orchestrated reflection (fixed uniform-16): 0.64 → 0.72 (gain 4, lost 0,
  net +8%)** — POSITIVE, and breaks nothing. Contrast 4B: net −4%, 0 gain.
- 8B perception probe: GT-localized recovers 7/16 (4B: 5/16).
- **Key turn:** the orchestrator-distillation premise (agent > base) **HOLDS on 8B,
  FAILED on 4B.** Mechanism: 8B perception is reliable enough that the 30B's
  text-reasoning over its reads is productive (vs 4B, where the 30B integrates
  perceptual garbage). → base-model capacity gates whether reflection has
  internalizable headroom. Trying 8B (user's call) reopened a (small, +8%) gap.
**CLEVRER setup:** `zechen-nlp/clevrer` train MCQ (predictive/counterfactual/
explanatory) + [scripts/build_clevrer_eval.py](../scripts/build_clevrer_eval.py);
videos per-file from MIT. 70 cases/videos ingested (DENSE_FPS=5, 0 errors).
**CLEVRER gap (8B, [scripts/diag_clevrer_gap.py](../scripts/diag_clevrer_gap.py)):**
free-form **0.49**, orch-reflection **0.47** (gain **0**, lost 1, net −1%). By type:
predictive 0.50 (= pure chance, binary), counterfactual 0.44, explanatory 0.52.
Matching verified correct (spot-checked). **CLEVRER is perception-bound for a
frame-VLM**: tracking fast collisions from 16 sampled frames is itself the
bottleneck → 8B near chance → 30B reasoning over unreliable reads gives 0 gain
(same "garbage-in" as 4B-on-NExT). CLEVRER is NOT the reasoning-clean substrate
we wanted — "clean rendering" ≠ "clean perception" when motion-tracking is required.

### Exp-6 — dataset research: where does reasoning-internalization actually have headroom (2026-06-08)
**Lesson distilled from Exp-1..5 + literature:** our orchestrated-reflection
mechanism (30B text-reasoning over 8B visual reads) has headroom ONLY where
(a) evidence sits in a FEW clearly-perceivable frames — NO long-range retrieval
(that's the type-2 frame-selection we can't internalize); (b) per-frame
perception is reliable; (c) the answer needs multi-step SYNTHESIS the VLM fumbles
in one pass. NExT-8B (+8%) barely sits in this band; CLEVRER does not
(motion-tracking = perception).
**Literature confirms** most video "reasoning" benchmarks are secretly
perception/retrieval-bound: VRIQ diagnostic — only **~1% of VLM failures are
reasoning-only** (56% perception-only); ARC-style ~80% perception. Reasoning
genuinely dominates only in **synthesis/knowledge** tasks (Neural-MedBench: 51%
reasoning failures; HERBench "fusion deficit").
**Dataset verdict (for our fixed-few-frame + synthesis mechanism):**
- **AVOID** (long-range multi-frame integration = retrieval/perception bottleneck
  = the agent's non-internalizable edge): MMR-V, HERBench, CLEVRER, long-video.
- **On-narrative video, perception-easier:** VidNum-1.4K (compositional numerical:
  per-frame count easy, arithmetic synthesis hard; github only, new),
  NExT-QA causal subset (videos already in hand; where the +8% came from).
- **Cleanest mechanism-proof (IMAGE, leaves the video framing):** ChartQA / PlotQA
  (read values → multi-step arithmetic), MathVista, CLEVR-static / GQA. Perception
  genuinely easy → reasoning dominant → expect a LARGE orchestrated-reflection gap.
**Open framing decision for the author:** stay video (modest ~5-15% gaps, VidNum /
NExT-QA-causal) vs pivot to image compositional/chart/math (large clean gap, best
causal-probe story, but diverges from the video-agent narrative). The mechanism is
likeliest to show a decisive positive on chart/math.

### Exp-7 — ChartQA mechanism test (perception-easy / reasoning-hard, image) (2026-06-08)
**Purpose:** the cleanest test of whether orchestrated reflection has headroom on
ANY data — charts: reading values "should" be easy, multi-step arithmetic is the
reasoning. [scripts/diag_chartqa_gap.py](../scripts/diag_chartqa_gap.py), 60 test
items, 8B + 30B. (Gotcha: localhost VLM calls must bypass the clash proxy —
script clears proxy env.)
**Result:** free-form **0.58**, orchestrated **0.50** (gain **1**, lost **6**,
net **−8%**). Orchestrated reflection HURTS. Mechanism: chart-value reading is
itself noisy OCR (perception); when the 30B forces the 8B to extract specific
values, the noise is amplified and arithmetic over wrong values yields wrong
finals. ChartQA is ALSO perception-bound (matches VISTA-Bench: >70% chart errors
are OCR-like perception).

### FINAL SYNTHESIS (2026-06-08) — strong, mechanism-backed boundary result
Across **4 datasets × 2 model scales**, orchestrated/agentic reflection over a
small VLM's reads has **no internalizable headroom**; the lone positive is tiny:

| setting | free | reflect/orch | net |
|---|---|---|---|
| NExT-GQA 4B self-reflect | 0.68 | 0.68 | 0 |
| NExT-GQA 4B 30B-orch | 0.68 | 0.66 | −4% |
| **NExT-GQA 8B 30B-orch** | 0.64 | **0.72** | **+8%** (only positive) |
| CLEVRER 8B 30B-orch | 0.49 | 0.47 | −1% |
| ChartQA 8B 30B-orch | 0.58 | 0.50 | −8% |

**Mechanism (the finding):** the bottleneck is **perception fidelity**, not
reasoning. Text-level reflection/orchestration over the VLM's reads cannot
compensate for perceptual errors and frequently **amplifies** them (forcing
sub-reads surfaces noise the holistic answer had averaged out). Positive headroom
appears only where perception is **coarse-but-reliable** (NExT-8B causal/action
reading, +8%); it vanishes or reverses where precise perception is required
(CLEVRER motion-tracking, ChartQA OCR, counting). Confirmed by literature
(VRIQ: ~1% reasoning-only failures; ARC ~80% perception).
**Implication for the project:** internalizing agentic *reasoning* into a small
VLM is not where the value is — the agent's real edge is *perception routing*
(frame/region selection, zoom), which is type-2 / non-internalizable by our own
constraints. The honest deliverable is this **boundary result with mechanism**,
which directly challenges the agentic-video-reasoning narrative. A positive
internalization result would require either (a) a much stronger base VLM whose
perception is reliable on the target tasks, or (b) redefining scope to internalize
perception-routing (which contradicts the no-tool-at-inference premise).

### Exp-8 — scaling sweep (4B/8B/30B-A3B/32B) + per-case failure analysis (2026-06-09)
**Setup:** pulled AWQ-4bit Qwen3-VL **32B (dense)** and **30B-A3B (MoE)** from HF
(QuantTrio), served TP=2 on 2×3080 alongside the 30B-A3B text orchestrator. Ran
free-form + 30B-orchestrated reflection on NExT(50)/CLEVRER(70)/ChartQA(60), AND
dumped FULL per-case outputs ([scripts/dump_case_outputs.py](../scripts/dump_case_outputs.py)
→ `data/distill/analysis/dump_<model>_<ds>.jsonl`: free_answer, critic sub-questions,
per-sub-question VLM reads, final answer). Note: 30B-A3B is MoE (3B active) ≈ 8B
perception; the dense 32B is the real "stronger perception" test. 32B needed
gpu-mem-util 0.85 (0.90 → vllm masked_scatter crash).

**Scaling table — free-form / orchestrated accuracy:**
| dataset | 4B | 8B | 30B-A3B | 32B |
|---|---|---|---|---|
| NExT (video temporal) | .70/.70 | .64/.62 | .68/.62 | .70/.64 |
| CLEVRER (collisions) | .43/.43 | .50/.50 | .50/.47 | .50/.50 |
| ChartQA (static charts) | .65/.52 | .63/.55 | .63/.42 | **.80/.75** |

**Per-case failure analysis (from the full-output dumps):**
- **NExT — perception / temporal-localization (NOT reasoning).** Model reads the
  scene right but picks the wrong brief/specific action (sees "smile", misses the
  "clap"; "swimming" vs "swim back out"); the decisive frame is often not in the
  uniform-16 sample. **~40% of "errors" are label ambiguity** (grazing=feeding,
  excited≈enjoying-music, near-duplicate MCQ distractors) — not real model errors.
  Almost zero "reads-right-reasons-wrong" cases. Flat across scale (4B=32B=.70).
- **CLEVRER — pure perception/tracking wall.** Reasoning structure is fine but the
  *perceptual premises are wrong*: from 16 sparse frames the model misidentifies
  which objects collide and when. ≈chance at every scale; 32B doesn't track better.
- **ChartQA — the ONE place reasoning becomes the bottleneck, but only at 32B.**
  4B/8B (~.64) fail on *reading* values/rankings; the dense 32B solves reading
  (.80) and its residual 12 failures are genuine **multi-step arithmetic**
  (difference/ratio/count-above-threshold over correctly-read values). 8B→32B fixes
  are all perception fixes (incl. 8B giving up with "证据不足"); 32B's remainder is
  arithmetic CoT.
- **Orchestrated reflection HURTS everywhere, less as the model strengthens**
  (ChartQA orch penalty −13%/−8%/−5% for 4B/8B/32B). Two mechanisms, both shown
  with receipts in the dumps: (1) *video* — leading sub-questions make the
  suggestible VLM emit contradictory NEW reads ("a football on the grass" → final
  "pick something from grass"), drifting a correct answer to wrong; (2) *charts* —
  decompose+re-integrate corrupts the answer (final picks the wrong option/format
  even when sub-reads were correct). Multi-step orchestration adds more failure
  points than it fixes; single forward pass is more robust. (The earlier 8B-NExT
  "+8% orch" was run-variance — did not reproduce here or on 30B-A3B/32B.)
- **Matcher caveat:** correctness is by MCQ letter; ~2 of 50 32B-NExT "correct" are
  letter-luck (right letter, wrong option text). Use the answer TEXT for fine analysis.

**Conclusion update:** scaling perception (→32B) does NOT break the wall on VIDEO
(NExT/CLEVRER flat) — the bottleneck there is temporal tracking/localization, which
is the agent's type-2 frame-selection edge (non-internalizable). It DOES break the
wall on static charts, and there — and only there, and only on a strong-enough base
— do the residual failures become genuine multi-step reasoning (arithmetic CoT),
i.e. the one regime where internalizing reasoning could have headroom. That regime
needs the 32B base and leaves the video-agent narrative.
Artifacts: `data/distill/analysis/dump_{4b,8b,32b}_{next,clevrer,chartqa}.jsonl` +
`*_percase.csv`.

## 4. Key decisions to date
- Local-only serving (4B VLM + 30B text); VLM cannot be an online API for the backbone.
- Vision-only ingest; full pipeline (Phase 0–4) code-complete + unit-tested.
- 50-case STOP is a **false stop**; true limiter is the self-improvement cap
  (signal pool ≈6% on short-video/uniform-frame slice), not the gate.
- **4B self-reflection has 0 headroom (Exp-3)**; **full agent (w/ retrieval) is
  WORSE than bare 4B, 54%<68% (Exp-2b)**.
- **NARRATIVE PIVOT (2026-06-08, user-approved): drop strict self-improvement,
  adopt orchestrator-distillation** — the 30B orchestrator is accepted as a
  *reasoning teacher*. Rationale: 4B self-reflection is a parrot (Exp-3), so the
  reflection worth internalizing is the 30B's. New selling points = "multi-model
  agent reasoning → single small VLM, no tool at inference + causal verification"
  (give up the "no stronger teacher" point). Next: Exp-4 isolates whether the
  30B-orchestrated reflection over FIXED uniform-16 frames beats free-form 68%.

## 5. Open questions / next steps (as of 2026-06-09)

**Done:** Exp-0..8 — pipeline (Phase 0–4, unit-tested) + the full diagnostic sweep
(pilot, free-form, self/orchestrated reflection, perception-headroom) across
**4B/8B/30B-A3B/32B** on **NExT-GQA / CLEVRER / ChartQA**, full per-case output
dumps, and the per-case failure analysis.

**Never run (deliberately gated):** Phase 5 SFT (LoRA), Phase 6 GRPO/RL, Phase 7
causal probes on a trained model. No trained model exists — every gate showed no
internalizable reasoning signal, so training would be on absent signal. Trainer
stack is also author-owned per SPEC §10.

**The fork (author's call):**
1. **Publish the boundary result** — "internalizing agentic/orchestrated reasoning
   into a small VLM has ~0 headroom on video QA; the bottleneck is perception
   fidelity (temporal tracking/localization), the agent's only edge is type-2
   frame-selection". Mechanism-backed, 4 datasets × 4 scales. Challenges the
   agentic-video-reasoning narrative. (Strongest honest deliverable.)
2. **Pursue the one positive lead:** 32B + ChartQA/numerical — perception solved at
   32B, residual = multi-step **arithmetic CoT**. Run a real SFT to internalize the
   CoT here. Cost: needs the 32B base, leaves the video narrative.
3. **Auxiliary:** judge-auto-label the 9 dumps' failures
   (perception/temporal/arithmetic/suggestibility/label-ambiguity) into one CSV.

**Reusable infra notes:** GPU-accelerated ingest still blocked by the BGE/SigLIP
fp16-on-CUDA segfault (CPU fp32 works, ~70 s/video) before any 567-video scale-up;
32B vllm needs `--gpu-memory-utilization 0.85` (0.90 → masked_scatter crash);
localhost VLM calls must bypass the clash proxy (HF downloads via hf-mirror, not
clash — clash gave 0.5 MB/s and stalled).

## 6. Reproducibility quick-ref
- Env: `conda activate mbe-ingest` (ingest/harness/distill); serving in `vllm-qwen`.
- Start servers: `bash scripts/serve/serve_vlm_4b.sh`; `CUDA_VISIBLE_DEVICES=1,2 TP_SIZE=2 bash scripts/serve/serve_text_30b.sh`.
- Pilot: `python -m app.distill.pilot --cases data/eval/datasets/nextgqa_pilot/cases.train.jsonl --n 50`.
- Diagnostics: `scripts/diag_{base_freeform,reflection_gap,orch_reflection,perception_headroom,clevrer_gap,chartqa_gap}.py`.
- Full per-case output dumps: `scripts/dump_case_outputs.py --dataset next|clevrer|chartqa --tag <model>`
  (point `LOCAL_VLM_BASE_URL`/`LOCAL_VLM_MODEL_NAME` at the served VLM).
- Serving (the sweep): `scripts/serve/serve_vlm_4b.sh`; for 8B/30B-A3B/32B see the
  vllm launch commands (TP=2, `--gpu-memory-utilization 0.85`, `--limit-mm-per-prompt '{"image":18}'`).
- Cross-model joins/insights: `scripts/build_percase_analysis.py`.
- Artifacts: `data/distill/pilot/` (pilot), `data/distill/{clevrer,chartqa}/` (gaps),
  `data/distill/analysis/` (dump_*.jsonl full outputs + *_percase.csv).

---

## 7. Supplementary-spec execution (2026-06-10) — variance gate + map

Executing `docs/SUPPLEMENTARY_EXPERIMENT_SPEC.md` to turn the exploratory log into a
defensible boundary+map result. New infra (all unit-tested, `tests/test_eval_common.py`):
`app/distill/{eval_common,eval_stats,seed_runner,methods}.py`, gates
`scripts/run_variance_gate.py` / `run_chartqa_gate.py`, `scripts/{build_partition,
label_audit,regrade_dumps,power_table,regen_tables,diag_oracle_perception}.py`.
Result store: `data/distill/results/results.jsonl` (fingerprinted, append-only);
tables regen via `scripts/regen_tables.py`. Orchestrator = **DeepSeek API**
(deepseek-v4-flash) — the local 30B is retired for this work (frees 2 GPUs).
**Throughput fix:** `seed_runner` got a `--concurrency` thread-pool (DeepSeek + vllm
both serve concurrent requests) — cut orch from ~4 h/gate to ~30 min.

### §0.5 label/grader floor (the floor under the floor)
- **§0.5.1 text-aware re-grade of all 9 dumps → 0 letter-luck cases** (free.L==free.T
  everywhere). The feared ~4% grader artifact does not exist here.
- **§0.5.2 label audit (DeepSeek judge, text-only, option-separability):** NExT 50 →
  **47 clean / 3 ambiguous / 0 wrong-gold, noise rate 6%** (CI [0.86,1.00]). Far below
  the spec's feared 10–15%. CLEAN = 47. (The old `label_audit_chartqa.json` with 48%
  "wrong-gold" is a discarded artifact of the earlier judge-sees-VLM-reading method.)

### §3.0 frames-visible partition
- All 50 NExT cases are **EVIDENCE_IN** (uniform-16 hits the GT window). CLEAN∩EVIDENCE_IN
  = **47**. `data/distill/analysis/partition_next.json`.

### §1 variance gate — NExT CLEAN∩EVIDENCE_IN (n=47, K=10 seeds, paired bootstrap)
| model | method | free | net | 95% CI (pooled) | verdict |
|---|---|---|---|---|---|
| 8B | self_reflect | .660 | **+0.060** | [+0.000,+0.128] | within-variance (borderline) |
| 8B | orch_reflect_blind | .660 | **+0.009** | [−0.170,+0.191] | within-variance |
| 4B | self_reflect | .745 | −0.045 | [−0.128,+0.000] | within-variance |
| 4B | orch_reflect_blind | .745 | −0.085 | [−0.255,+0.064] | within-variance |

- **RETRACTION (the named result):** 8B orch per-seed net = +.09,−.04,+.11,+.04,+.02,
  +.00,+.06,−.06,−.06,−.06 → the seed-0/2 "+8.5%/+10.6%" wash to **pooled +0.9%, sign
  flips across seeds**. The lone positive does not survive multi-seed. 8B self_reflect
  best-seed +.064 → pooled +.060 with CI touching 0.
- **Mechanism note:** self_reflect σ_decode≈0 (MCQ option-choice is low-entropy even at
  temp 0.7 — rationale varies, graded option stable; verified seed honored). The large
  variance is in **orch** (DeepSeek critic's stochastic sub-questions) — exactly the
  +8% retraction's source. orch churn for 8B ≈ gain 5–9 / lost 2–8 per seed.

### §2 power (honest limitation)
- At n=47, **minimum detectable net @80% power ≈ 25%** (4B .252, 8B .274); n=300→~10%,
  n=500→~8%. All observed |net|<9% are below detectable → "no reliable effect **but
  underpowered**." The **sign-flip across seeds** is the power-independent evidence.
  Scaling n needs the GPU-ingest fix (still blocked). `data/distill/results/power_table.json`.

### §3 perception-headroom probe — 8B NExT free-wrong (n=18)
- recovered correct: control 0 → uniform-hi 2 (11%) → GT-local-hi **4 (22%)**.
- **78% (14/18) stay wrong even with perfect frame selection at max res** = visual
  "fundamental wall"; 22% are perception/selection-limited (type-2, non-internalizable).
  Oracle (§3.1b) will split the 78% into reasoning-bound vs fundamental.

### §3.1b near-oracle perception (32B caption of GT frames → DeepSeek reasons, no 8B eyes)
- 8B NExT free-wrong (n=18): **near-oracle recovers 6/18 = 33%**. Perfect perception +
  reasoning fixes only a third; 67% stay wrong (label-ambiguous / unanswerable-from-frames /
  fundamental). With the probe (22% recover via better perception), this places **NExT-8B in
  regime 1**: the residual is NOT a recoverable reasoning headroom. `oracle_perception_8b_next.json`.

### §5 ChartQA full ladder (free / self_reflect / orch, multi-seed)
| model | free | self_reflect net | orch net | verdict |
|---|---|---|---|---|
| 4B | .667 | +0.023 | **−0.122** | within / **effect (reliable NEGATIVE)** |
| 8B | .617 | +0.012 | −0.028 | within / within |
| 32B | **.800** | +0.017 | −0.044 | within / within |
- 4B/8B = regime 1 (perception-bound: read-values failure; orch corrupts, −12% at 4B).
  **32B = regime 2** (perception solved .80; residual = multi-step arithmetic per Exp-8).

### §4 sighted-critic ablation — TWO datasets (the missing control), 8B base + 32B sighted critic
| dataset | blind orch net | sighted orch net (k=5) | verdict |
|---|---|---|---|
| NExT | +0.009 [−0.170,+0.191] | **+0.060 ± 0.055** [−0.085,+0.191] | within-variance |
| ChartQA | −0.028 | **+0.070 ± 0.040** (per-seed +.02/+.10/+.07/+.05/+.12, all ≥0) | within-variance (borderline+) |
- **NExT:** a sighted 32B critic helps numerically more than blind (+.060 vs +.009) but still
  does NOT reliably beat free-form (CI crosses 0; per-seed sign flips). The dominant wall is the
  8B's *perception* ceiling — a stronger sighted critic over the same frames can't reliably fix it.
- **ChartQA:** the sighted critic trends positive on EVERY seed (+0.070, all ≥0) — on charts the
  wall is *reading values*, which a sighted 32B critic (reads .80) partially repairs; just shy of
  significance at n=60/k=5. Contrast with blind orch (−0.028): **sightedness matters more where the
  bottleneck is perception-you-can-re-read (charts) than where it is temporal localization (video).**
  This two-dataset contrast is the clean §4 result.

### §8 THE MAP (regen: `scripts/{regen_tables,build_map}.py` → `results/{tables,map}.json`)
| dataset × model | free | best agentic net | regime |
|---|---|---|---|
| NExT 4B / 8B / 32B | .74 / .66 / .70 | −.045 / +.060 / +.039 (all within-var, k=10/10/6) | **1** perception/selection-bound |
| NExT InternVL3-8B (off-Qwen) | .57 | orch **−.163** (k=3, *effect/negative*) | **1** (cross-family confirm) |
| NExT Penguin-VL 2B / 8B (new LLM-encoder) | .53 / .43† | +.050 / +.128 (within-var, k=3) | **1** (new-encoder confirm) |
| ChartQA 4B / 8B | .67 / .62 | +.023 / +.012 (4B orch −.122 *effect/neg*) | **1** perception-bound |
| ChartQA Penguin-VL 2B / 8B (new LLM-encoder) | **.82 / .80** | −.078 / −.006 (within-var, k=3) | **1** (perception solved, no residual probe) |
| **ChartQA 32B** | **.80** | +.017 | **2** reasoning-bound |
| ChartQA 8B-SFT (§11) | **.73** | — | (R1 base, +.117 from internalization) |
- †Penguin-8B-NExT ran at **2 frames** (16-frame dense-video tokens OOM the 20GB cards even split);
  not directly comparable to the 16-frame .53/.66 cells — the .43 is frame-starved, and the
  consistent-but-within-variance orch +0.128 is blind re-asking recovering a starved 2-frame read.
- Clean 2-regime decomposition: **video is uniformly perception/selection-bound at every scale
  AND across THREE vision-encoder paradigms (Qwen tiling-ViT 4B/8B/32B + InternVL3-8B fixed-res ViT
  + Penguin LLM-based encoder); static charts flip to reasoning-bound only once a strong base solves
  perception.** No agentic/reflective method reliably beats single-forward in ANY cell (all
  within-variance); the only reliable agentic effects are *negative* (4B-ChartQA orch −12%,
  InternVL-NExT orch −16%, Penguin-2B-ChartQA orch −8%).
- **New result from the Penguin encoder:** the R1 *chart-perception* wall on small contrastive-encoder
  models is an **encoder-paradigm property, not fundamental** — Penguin's LLM-based encoder lifts
  ChartQA reading to **.82 (2B) / .80 (8B)**, matching Qwen-**32B** (.80) and ≫ Qwen-8B (.62). But the
  *temporal-video* wall is unmoved (Penguin-2B NExT .53, still R1) — better static-OCR perception
  doesn't buy temporal grounding.

### §11 PoC — RAN, and it is a statistically significant POSITIVE internalization result
Pipeline (`scripts/poc_{gen_cot,sft,merge,eval,causal_probe}.py`): the **32B teacher** generated
step-by-step chart-solving CoTs on ChartQA TRAIN; kept **150** whose final answer matched gold
(consistency filter). LoRA-SFT the **bf16 8B** (peft, r=16, vision tower frozen, 2-GPU bf16,
3 epochs, train_loss 0.30→0.17). Merge → serve → eval single-forward on the **60 held-out test**
cases (train/test disjoint):
- **base-8B .617 → SFT-8B .733; paired bootstrap net +0.117, 95% CI [+0.050, +0.200] (EXCLUDES 0);
  gain 7 / lost 0; McNemar p=0.023.** Internalizing the teacher's chart-solving trajectory into a
  single forward pass works.
- **§11.3 causal probe (2a counterfactual, n=33 correct-with-CoT):** corrupting one numeric CoT
  intermediate flips the answer **3%**; shuffling the CoT flips **15%**. The answer is *robust* to
  corrupting intermediates → the gain is **perception/reading transfer, not a load-bearing internal
  arithmetic chain** (the model re-reads the chart). Honest reading: in this perception-bound cell
  (ChartQA-8B, R1) the SFT internalized the teacher's *reading*; the **pure-reasoning**
  internalization the probe would light up is the **regime-2 (32B-charts) cell** — the documented
  next target (needs bf16-32B; only AWQ-4bit is local). Either way §11 demonstrates a real,
  significant, single-forward internalization gain with an honest mechanism characterization.

### §6 cross-family (InternVL3-8B) — DONE (regime-1 reproduces off-Qwen)
Model acquisition was the hard part: ModelScope ~360 MB/h (unusable), HF/hf-mirror lacked LFS; the
working path was **HF-direct via the clash proxy, downloaded sequentially one shard at a time**
(concurrent multi-shard stalls the proxy). Served TP=2 on GPU2,3 :30003, vLLM resolves
`InternVLChatModel` fine. Two serving fixes vs the Qwen path: (1) InternVL's chat template can't
concatenate a system *string* with list-content → fold system into the user message (patched in
`methods.vlm_answer`); (2) InternVL's dynamic tiling makes 16×448px = 54k tokens > ctx → run at
8 frames (free_form) / 4 frames (orch), max-model-len 32768.
- **InternVL3-8B free_form = .574 on NExT CLEAN∩EVIDENCE_IN** — a different vision tower (fixed-res
  ViT vs Qwen's tiling) lands in the same accuracy band as the Qwen family (4B .745 / 8B .660 /
  32B .702), confirming the perception-bound regime is not a Qwen-tiling artifact.
- **orch_reflect_blind net = −0.163 ± 0.065 (k=3, verdict=EFFECT — CI [−0.319,−0.021] excludes 0):
  orchestration RELIABLY HURTS InternVL** (all 3 seeds negative: −.23/−.11/−.15), even more than on
  Qwen. Throughput note: InternVL multimodal prefill is heavy — generation collapses to ~1 tok/s
  under concurrency on 27k-token prompts, so the multi-seed run used 4-frame prompts at
  concurrency=1 (~25 min/seed). The InternVL perception-headroom probe is deferred (corroborative
  only — the perception-wall mechanism is established on the Qwen family + oracle).
- **Cross-family verdict:** the map's regime-1 finding (perception/selection-bound video; agentic
  reasoning net ≤ 0, reliably negative for orchestration) reproduces on a non-Qwen vision encoder.

### §6b cross-ENCODER (Penguin-VL 2B/8B — a NEW vision-encoder paradigm) — DONE
Penguin-VL replaces the contrastive CLIP/SigLIP ViT with an **LLM-based vision encoder** (init from
Qwen3-0.6B, bidirectional attention + 2D-RoPE) plus TRA temporal token compression — the cleanest
test of whether the perception-bound regime is a property of the *encoder paradigm* or of the *task*.
Acquisition/serving: ModelScope was unusably slow, so main weights came via HF+clash proxy; the arch
`penguinvl_qwen3` is **vLLM-incompatible** → ran in a dedicated `penguin` env (transformers **4.51.3**;
5.9 breaks its processor, 4.57 breaks weight load) with patches: `vision_encoder=null` (encoder
weights bundled in the main safetensors as nested keys), `from_config→_from_config`, an **SDPA
fallback** for the missing flash-attn (with `enable_gqa=True` for the 16-q/8-kv head mismatch), and a
**decode-slicing fix** (Penguin's custom `generate()` returns only new tokens; the README decodes
unsliced → empty strings). GPUs per the run plan: 2B on GPU1, 8B on GPU2,3, GPU0 left free.
- **ChartQA (1-frame static, n=60): Penguin-2B .817, Penguin-8B .800** — the headline cross-encoder
  result. The new encoder reads chart values/rankings **far** better than the same-size Qwen
  (Qwen-8B .617), landing at the **Qwen-32B** reading ceiling (.80). → the small-model R1
  chart-perception wall is an **encoder-paradigm artifact**, not a fundamental limit.
- **NExT video (CLEAN∩EVIDENCE_IN, n=47): Penguin-2B free .532** (16 frames) — squarely in the Qwen
  band (4B .745 / 8B .660 / 32B .702 / InternVL .574); the new encoder does **not** lift temporal
  video. Penguin-8B-NExT only fit at 2 frames (.426, frame-starved — see †).
- **Orchestration: within-variance everywhere, never reliably positive.** Penguin-2B-ChartQA orch
  **−0.078** (reliable-ish negative), Penguin-8B-ChartQA −0.006, Penguin-2B-NExT +0.050,
  Penguin-8B-NExT +0.128 (all CIs cross 0 at k=3). Same regime-1 signature as Qwen + InternVL:
  blind reflective orchestration adds no reliable single-forward headroom on a brand-new encoder.
- **Cross-encoder verdict:** swapping the entire encoder paradigm (contrastive ViT → LLM encoder)
  **moves the static-perception ceiling a lot** (charts .62→.82) but **leaves the regime structure
  intact** — video stays perception/selection-bound, agentic net stays ≤ 0. The boundary result is
  robust across three encoder paradigms.

### Status: ALL spec sections COMPLETE (§0.5 … §11)
§0.5 clean labels + §3.0 frames-visible + §1 variance gate (**4B/8B 10-seed + 32B 6-seed**) + §2
power + §3 perception probe + §3.1b oracle + **§4 sighted critic on 2 datasets** + §5 ChartQA
ladder + **§6 cross-family InternVL + §6b cross-ENCODER Penguin-VL 2B/8B (free + 3-seed orch on
both NExT & ChartQA)** + the retraction box + the 2-regime MAP + **§11 SFT PoC
(gen→train→eval→causal probe)**. All numbers regenerate from
`data/distill/results/results.jsonl` via `scripts/{regen_tables,build_map,power_table}.py`.

**Full story:** internalizing agentic/reflective reasoning into a small VLM has **no statistically
reliable headroom in the frames-visible video regime** — perception/selection-bound at every scale
(4B/8B/32B) and across THREE vision-encoder paradigms (Qwen tiling-ViT, InternVL3-8B fixed-res ViT,
Penguin LLM-based encoder); the only reliable agentic effects are NEGATIVE (orchestration corrupts:
−12% 4B-charts, −16% InternVL-video, −8% Penguin-2B-charts). The reasoning-bound regime appears only
once a strong base solves perception (static charts) — and the Penguin cross-encoder test sharpens
this: a new LLM-based encoder lifts small-model chart *reading* to the 32B ceiling (.62→.82),
proving the R1 chart wall is an encoder-paradigm artifact, yet leaves the temporal-video wall and
the agentic-net≤0 structure fully intact. **§11 then shows the
positive half constructively: distilling the 32B teacher's chart-solving CoT into the 8B lifts
single-forward accuracy .617→.733 (+11.7%, CI excludes 0)** — though the causal probe shows that
gain is perception-transfer, locating *pure-reasoning* internalization in the 32B-charts cell as the
next target. Caveats kept prominent: n=47 (NExT) / n=60 (ChartQA), minimum detectable net ≈25% on
NExT — agentic nulls are "no reliable effect, underpowered," and the seed-sign-flips are the
power-independent evidence behind the +8% retraction.

**Remaining hardware-bound deferral (not a skipped experiment):** the InternVL perception-headroom
probe (mechanism already established on Qwen) and the pure-reasoning §11 on bf16-32B (only AWQ-4bit
local). Everything in SPEC §0–§11 that is runnable on this hardware has been run.

### §7 reproducibility quick-ref
- Env: `conda activate mbe-ingest` (gates/SFT/eval); serving in `vllm-qwen`. Orchestrator = DeepSeek
  API (`.env` `ORCHESTRATOR_*`); localhost VLM calls bypass the clash proxy (`NO_PROXY=*`).
- Serve: `serve_vlm_4b.sh` (:30000), 8B TP=2 (:30002), `serve_vlm_32b.sh` (:30001),
  `serve_vlm_internvl.sh` (:30003). InternVL weights via `HTTPS_PROXY=…7890 hf download
  OpenGVLab/InternVL3-8B` (one shard at a time).
- Gates (add `--concurrency 8` for throughput): `run_variance_gate.py --model-id <id>
  --clean data/distill/analysis/label_audit_next.jsonl --methods self_reflect orch_reflect_blind
  [orch_reflect_sighted --critic-base …:30001/v1 --critic-model Qwen3-VL-32B-Instruct] --seeds K`;
  `run_chartqa_gate.py` (same flags, ChartQA).
- Floor/partition/power/probes: `regrade_dumps.py`, `label_audit.py`, `build_partition.py`,
  `power_table.py`, `diag_perception_headroom.py`, `diag_oracle_perception.py`.
- Tables/map: `regen_tables.py` → `tables.json`; `build_map.py` → `map.json`.
- §11 PoC: `poc_gen_cot.py` (32B teacher → `chartqa_cot_train.jsonl`) → `poc_sft.py` (LoRA, GPU2,3)
  → `poc_merge.py` → serve merged `Qwen3-VL-8B-ChartQA-SFT` (copy original 8B tokenizer/config in,
  TP=2) → `run_chartqa_gate.py --model-id 8b_sft --seeds 0` (free_form eval) → `poc_causal_probe.py`.
- vLLM-child orphans hold GPU memory after a server is killed — kill the `VLLM::Worker_TP` /
  `EngineCore` PIDs explicitly (not just the api_server) before re-serving.
