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

**Active pivot (2026-06-08):** moving toward **Option B / reflection** — hold
frames constant (same uniform-16 for tool-agent and free-form) so the controlled
variable is the agent's *reflection* (self-re-examine / revise), and test whether
that reflection ability can be internalized. See Exp-3.

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

## 5. Open questions / next steps
1. Exp-3 reflection gap number → decide B viability.
2. If B: design reflection-trajectory generator (multi-turn 4B self-reflection,
   tool-free, fixed frames) as the trajectory source; rewrite reflection→single-
   forward CoT; consistency gate (reflect-right ∧ free-wrong); SFT + causal probes
   (does internalized single-forward recover the multi-turn reflection gain?).
3. Strict filter for distillation: relax to answer-correct + grounding-hit (drop
   product citation/agent-loop tags) — recovers the few process-dropped candidates.
4. GPU-accelerated ingest (SigLIP segfault) before any 567-video scale-up.
5. Training side (SFT LoRA / GRPO) stays gated until a real signal pool exists.

## 6. Reproducibility quick-ref
- Env: `conda activate mbe-ingest` (ingest/harness/distill); serving in `vllm-qwen`.
- Start servers: `bash scripts/serve/serve_vlm_4b.sh`; `CUDA_VISIBLE_DEVICES=1,2 TP_SIZE=2 bash scripts/serve/serve_text_30b.sh`.
- Pilot: `python -m app.distill.pilot --cases data/eval/datasets/nextgqa_pilot/cases.train.jsonl --n 50`.
- Diagnostics: `scripts/diag_base_freeform.py`, `scripts/diag_reflection_gap.py`.
- Artifacts under `data/distill/pilot/` (trajectories/, cot/, *_report.json, *_diag.json).
