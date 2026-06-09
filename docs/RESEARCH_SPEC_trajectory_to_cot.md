# SPEC: Agent-Trajectory-to-CoT Internalization for Small VLM Visual Reasoning

> **Audience**: This spec is written for an AI coding agent (Codex) that will implement
> the pipeline incrementally. It assumes the existing **Mr. Big-Eye** codebase
> (FastAPI + LangGraph 14-tool long-video audiovisual QA agent) is available and working.
> Read the “Existing Codebase Contract” section before writing any code. Do NOT rebuild
> what already exists. The author is fluent in vLLM/SGLang deployment; model serving is
> out of scope for you to design — assume an OpenAI-compatible endpoint exists.

-----

## 0. Research Goal (read first, this constrains every design decision)

**Hypothesis**: A small VLM (Qwen3-VL-4B-Instruct) can internalize multi-step agentic
visual-reasoning *as single-forward Chain-of-Thought*, such that after fine-tuning it
reasons better in ONE forward pass (no tools), and we can **prove** the gain is
*generalizable reasoning* rather than *template memorization*.

**Two hard constraints that override convenience:**

1. **Self-improvement framing (no stronger teacher).** The agent that GENERATES training
   trajectories must use **the same 4B model** as its VLM backbone — NOT a remote large
   API. Using a larger model as teacher invalidates the research narrative. The only
   exception: a remote model MAY be used as a *rewriter* (trajectory→CoT text rewriting),
   because it produces no new visual capability, only reformats text. This distinction
   must be preserved in code (separate config keys, see §3).
1. **Only internalize “type-1” steps.** Agent trajectories contain two kinds of steps:
- **Type-1 (internalizable)**: reasoning a single-forward VLM can in principle execute
  itself — counting visible objects, comparing colors, reading the clock in frame,
  ordering events it can see. These map to a `[FRAME:t=]` or `[SLIDE:t=]` anchor.
- **Type-2 (tool-dependent, MUST be stripped)**: steps whose conclusion came from an
  external tool the target VLM does not have — RRF retrieval ranking, FTS5 keyword hits,
  temporal resolver decisions, grounding verification, cross-frame re-ID scores.
  Putting type-2 steps into the CoT teaches hallucination. The rewriter and the filter
  must enforce this split. See §4 and §5.

**What we are selling**: the *method* and the *internalizability boundary* (which reasoning
types internalize vs. only memorize), NOT a leaderboard number. Design experiments for
clean causal attribution, not for maximizing accuracy.

### 0.1 Positioning (the axis that defines this work — do not blur it)

The closest related work is **VideoTemp-o3** (Liu et al., 2026, Kwai-Keye). It is easy to
mistake this project for theirs; the distinction must be stated precisely. **The dividing
axis is whether tools are present AT INFERENCE — not how many models there are.**

|                                       |Old Mr-Big-Eye (prior system)         |VideoTemp-o3 (closest related)                                        |THIS PROJECT (target)                         |
|---------------------------------------|--------------------------------------|----------------------------------------------------------------------|----------------------------------------------|
|# of models                            |many (orchestrator + VLM + retrievers)|**1 VLM**                                                             |1 VLM                                         |
|Who decides scheduling                 |separate orchestrator LLM             |the VLM itself (emits `<tool_call>` tokens)                           |the VLM itself                                |
|**Tools actually invoked at inference**|**YES**                               |**YES** (still localize-crop-answer, external Crop module in the loop)|**NO** (pure single forward pass)             |
|Fate of the multi-step process         |executed at runtime                   |executed at runtime (merged into one model)                           |**compressed into weights; absent at runtime**|
|Verifies reasoning-vs-memorization     |no                                    |no (result-only rewards)                                              |**yes (causal probes, §7)**                   |

Read the third row across columns 2→3→4: **Old Mr-Big-Eye and VideoTemp-o3 are on the SAME
side** — tools are present at inference. VideoTemp-o3’s contribution is collapsing
multi-model collaboration into one model’s multi-turn self-scheduling; its “internalization”
means the VLM internalized *when/where to crop*, but the localize-crop-answer loop and the
external Crop module are still live at inference (their own Appendix C confirms it still
crops). **This project internalizes one level deeper**: the multi-turn self-scheduling is
moved from runtime into training, so at inference the model emits no `<tool_call>`, no
external Crop runs, and it answers in a single forward pass from only the frames it can see.

**Positioning sentence for the paper / related-work (use verbatim as the anchor):**

> “VideoTemp-o3 internalizes *when and where to crop* but retains the localize-crop-answer
> tool loop at inference. We instead study whether the multi-step reasoning itself can be
> internalized into a *single forward pass with no tools at inference*, and — crucially —
> whether the internalized reasoning is genuine generalization or template memorization, a
> question their result-only rewards cannot address.”

**Forbidden claim**: do NOT sell “single model” as the novelty — VideoTemp-o3 already owns
that. The novelty is exactly two things: (1) *no tool at inference / single forward pass*,
and (2) *causal verification that the internalized reasoning is real, not a template*.

-----

## 1. Existing Codebase Contract (DO NOT reimplement these)

The Mr. Big-Eye repo already provides everything below. Treat these as fixed APIs.

|Component                                                                         |Location                                                                   |Reuse as                                                |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------|
|Offline indexing (scene caption, dense frames, ASR, OCR, Chroma/FTS5)             |`app/preprocess.py`, `app/retrieval.py`, `app/text_assets.py`, `app/asr.py`|**Asset producer — unchanged**                          |
|LangGraph 14-tool agent loop                                                      |`app/graph.py`, `app/tools.py`                                             |**Trajectory generator** (only swap VLM backbone)       |
|VLM client (OpenAI-compatible / responses)                                        |`app/vqa.py`                                                               |**Extend** with a local-backbone client (§3)            |
|Eval harness + `failure_tags` + Soft-Waive                                        |`app/eval_harness.py`, `app/eval_fingerprint.py`                           |**Trajectory filter** (§5)                              |
|`verify_grounding` tool                                                           |`app/tools.py` #13                                                         |**Consistency-filter checker** (§5)                     |
|Citation protocol `[FRAME:t=]` `[TRANSCRIPT:t=]` `[SLIDE:t=]`                     |`app/vqa.py`, `app/graph.py`                                               |**Type-1/Type-2 anchor markers** (§4)                   |
|QA pairs                                                                          |`eval/audiovisual/questions.ready.jsonl`, NExT-GQA converters in `scripts/`|**Reusable** (clean source of (video, question, answer))|
|GraphState fields (`observer_notes`, `subject_registry`, `grounding_report`, etc.)|`app/graph.py`                                                             |**Trajectory provenance** (§4)                          |

**Tool taxonomy for the type-1/type-2 split** (derived from the 14-tool table). The rewriter
in §4 MUST use this mapping:

|Tool                                                                 |Internalizability             |CoT treatment                                                                                                                                                  |
|---------------------------------------------------------------------|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`segment_focus` (#8)                                                 |**Type-1**                    |Keep — its `observation` is a direct visual reading of frames near a `[FRAME:t=]` anchor                                                                       |
|`stitched_verify` (#10)                                              |**Type-1**                    |Keep — cross-window visual comparison the VLM can do if shown the frames                                                                                       |
|`answer_with_evidence` (#12) draft reasoning                         |**Type-1**                    |Keep the visual reasoning; strip the citation-bookkeeping                                                                                                      |
|`retrieve_video_evidence` (#1)                                       |**Type-2**                    |Strip the *retrieval act*; keep only “frame at t=X shows …” as a given premise                                                                                 |
|`retrieve_transcript_evidence` (#2), `search_transcript_keyword` (#3)|**Type-2 (excluded by scope)**|Strip entirely. VISION-ONLY: any answer that *requires* transcript content is out of scope — drop the whole case (see §1 scope)                                |
|`retrieve_slide_evidence` (#4)                                       |**Type-2 (excluded by scope)**|Strip entirely. OCR text is not a pixel-recoverable signal in vision-only; do not turn it into a premise. Cases whose answer requires OCR text are out of scope|
|`align_audiovisual_evidence` (#5)                                    |**Type-2 (excluded by scope)**|Strip entirely; audiovisual diff questions are out of scope in vision-only                                                                                     |
|`build_timeline` (#6)                                                |**Type-2**                    |Strip the timeline construction; keep only the final visual ordering if it is visually grounded in shown frames                                                |
|`retrieve_hypothesis_evidence` (#7), `expand_temporal_evidence` (#9) |**Type-2**                    |Strip                                                                                                                                                          |
|`assess_evidence_sufficiency` (#11), `verify_grounding` (#13)        |**Type-2 (meta)**             |Strip entirely — never appears in CoT                                                                                                                          |
|`search_user_memories` (#14)                                         |**Type-2**                    |Strip entirely                                                                                                                                                 |


> **SCOPE — DECIDED: VISION-ONLY (frames only).** The target single-forward VLM receives
> ONLY the retrieved frames (dense/scene JPEGs the agent surfaced). NO transcript, NO OCR
> text, NO audio — at trajectory generation, rewriting, training, AND inference. This is a
> fixed decision, not a tunable. Consequences Codex must enforce:
> 
> 1. Any reasoning step depending on transcript or OCR *text content* is type-2 → stripped.
> 1. A case whose CORRECT ANSWER fundamentally requires audio/transcript/OCR-text evidence
>    is **out of scope** → the whole case is dropped, not partially rewritten. Add a
>    **case-level scope filter** in Phase 2 that drops cases whose gold trajectory’s decisive
>    evidence came from tools #2/#3/#4/#5 (transcript/slide/audiovisual). Detect this from the
>    trajectory’s `failure_tags`/evidence provenance: if removing all type-2 text evidence
>    would make the answer underdetermined, drop the case.
> 1. `TRAIN_MODALITY` is kept as a single fixed value `frames_only`; do not implement other
>    branches. (Kept as a named constant only so the assumption is greppable in code.)
>    This collapses several “conditional type-1” tools into unconditional strips (see table
>    above) and is expected to RAISE pilot retention by removing text-dependent reasoning.

-----

## 2. Pipeline Overview

```
Phase 0  Local VLM backbone client          (small, new code in app/vqa.py)
Phase 1  Trajectory generation              (reuse agent loop, 4B as backbone)
Phase 2  Strict filtering                    (reuse harness, tighten gates)
Phase 3  Trajectory -> CoT rewriting         (NEW module)
Phase 4  Consistency filtering               (reuse verify_grounding + replay)
Phase 5  SFT (LoRA) on (frames, question, CoT, answer)
Phase 6  GRPO/DAPO on filtered prompts        (verifiable reward)
Phase 7  Evaluation: reasoning vs memorization (causal probes)
```

**Mandatory gate before scaling**: run Phase 0→4 on a **50-case pilot** and measure the
consistency-filter retention rate (§5.4). If retention <30%, STOP and report — the task is
too tool-dependent for 4B single-forward; the author must shrink the task subset. If ≥50%,
proceed to full scale. This pilot gate is the single most important checkpoint; implement
it as a first-class CLI command, not an afterthought.

-----

## 3. Phase 0 — Local VLM Backbone Client

Extend `app/vqa.py` with a backbone abstraction so the agent can run on a local model while
keeping the remote path for comparison. Do not break existing remote behavior.

**Requirements:**

- Add a `VLMBackbone` protocol with the existing methods: `generate_caption(image)`,
  `answer_question(question, frames, timestamps, ...)`, `stream_answer_question(...)`.
- Two implementations: `RemoteVLMBackbone` (existing logic, refactored behind protocol) and
  `LocalVLMBackbone` (OpenAI-compatible client pointed at the author’s vLLM/SGLang endpoint).
- Config keys (Pydantic Settings in `app/config.py`), all read from `.env`:
  - `AGENT_VLM_BACKEND` = `remote` | `local` — which backbone the AGENT uses to generate trajectories.
  - `LOCAL_VLM_BASE_URL`, `LOCAL_VLM_MODEL_NAME` — the served 4B model.
  - `REWRITER_API_BASE_URL`, `REWRITER_API_KEY`, `REWRITER_MODEL_NAME` — text rewriter (§4); MAY be a larger model, allowed by constraint #1.
  - `TRAIN_MODALITY` = `frames_only` — FIXED constant, vision-only scope (§1). Do not implement other branches.
- The orchestrator (text tool-calling model) is independent of the VLM backbone and keeps
  its existing `ORCHESTRATOR_*` config. For trajectory generation it is acceptable for the
  orchestrator to remain a capable tool-calling text model — it only decides tool order, it
  does not inject visual capability into the distilled CoT (which is VLM-only reasoning).
- **No model serving code.** Assume the endpoint is up. Provide a `scripts/check_local_vlm.py`
  that pings the endpoint and runs one caption + one multi-frame QA as a smoke test.

**Acceptance**: existing tests in `tests/test_vqa.py` still pass; `AGENT_VLM_BACKEND=local`
makes the agent answer using the served 4B model end-to-end on one smoke video.

-----

## 4. Phase 1 + Phase 3 — Trajectory Generation & CoT Rewriting

### 4.1 Trajectory generation (Phase 1)

- Reuse the agent loop unchanged. Run it over the reusable QA pairs with `AGENT_VLM_BACKEND=local`.
- Persist **full structured trajectories**, not just final answers. Capture, per case:
  - ordered list of `(tool_name, tool_args, tool_result_summary, state_delta)` from the
    LangGraph run (read from `messages` + the `Command(update=...)` deltas).
  - all `observer_notes`, final `draft_answer`, `grounding_report`, `subject_registry`,
    `candidate_timeline`, `audiovisual_candidate_matrix`.
  - the exact frame timestamps that were shown to the VLM (the `[t=X.Xs]` tags) and their
    image paths under `data/cache/{video_id}/frames_*`.
- Write to `data/distill/trajectories/{case_id}.json`. Bump a new `DISTILL_SCHEMA_VERSION`.

### 4.2 CoT rewriting (Phase 3, NEW module `app/distill/rewrite.py`)

Convert each filtered trajectory into a natural single-forward CoT. The rewriter LLM
(`REWRITER_*`) is given the trajectory plus strict instructions:

**Rewrite rules (enforce in the prompt AND validate in code post-hoc):**

1. Output a first-person visual reasoning chain as if the model is looking at the provided
   frames ONCE, with no tools.
1. **Keep only type-1 steps** per the §1 tool taxonomy. Each retained reasoning step must be
   anchorable to a `[FRAME:t=]` timestamp that is in the shown-frames set. `[SLIDE:t=]` and
   `[TRANSCRIPT:t=]` markers are forbidden in vision-only scope.
1. **Convert type-2 tool conclusions into premises, only if the premise is visible in the
   shown frames.** E.g. a retrieval step that surfaced the right scene becomes nothing (the
   frames are simply present); it must NOT become a sentence like “I searched and found…”.
1. **Never** write tool names, “I retrieved”, “the resolver decided”, confidence scores, or
   any number the VLM could not compute from pixels (no “match score 0.91”).
1. Strip all `assess_evidence_sufficiency` / `verify_grounding` content.
1. End with the verified final answer. For MCQ, end with the chosen option.
1. Keep the CoT concise; do not pad with restated question.

**Post-rewrite code validation** (reject the sample if violated):

- Regex-scan the CoT for forbidden tokens: tool names, “retriev”, “resolver”, “FTS”, “RRF”,
  “score”, “rank”, “I searched”, “match” + numeric. Configurable blocklist.
- Only `[FRAME:t=]` markers are permitted. Every one must exist in the shown-frames set
  (reuse `verify_grounding`’s marker-proximity check).
- `[SLIDE:t=]` and `[TRANSCRIPT:t=]` markers are forbidden — reject the sample if present.

Output: `data/distill/cot/{case_id}.json` = `{video_id, question, shown_frames:[t...], cot, answer, source_traj_hash}`.

-----

## 5. Phase 2 + Phase 4 — Filtering (the soul of the pipeline)

Bad filtering silently poisons everything downstream. Two filter stages.

### 5.1 Phase 2 — Strict trajectory filter (reuse harness, TIGHTEN gates)

The existing harness uses **Soft-Waive** (“answer correct ⇒ pass, keep process issues as
forensic tags”). That is correct for evaluation but WRONG for training-data selection,
because it admits “lucky-correct, bad-process” trajectories. For distillation, **disable
Soft-Waive** and require ALL of:

- final answer correct (rule-based match, or judge agreement for open-ended);
- `failure_tags` empty (no citation-kind error, no missing frame/slide evidence, no
  temporal/factual wrong-option flag);
- grounding actually hit (frame markers within tolerance of shown frames);
- trajectory length within a sane band (drop runs that hit the 8-tool cap or that looped).

Implement as a harness mode flag, e.g. `--distill-strict`, that flips Soft-Waive off and
enforces the conjunction above. Reuse `failure_tags` — do not invent a new tagger.

**Vision-only case-scope filter (Phase 2, also drop the case if):** the trajectory’s
decisive evidence came from transcript/slide/audiovisual tools (#2/#3/#4/#5), i.e. removing
all non-visual text evidence would leave the answer underdetermined. These cases are out of
scope (§1) — drop the whole case, do not try to rewrite it into a visual CoT. Keep a count
of dropped-as-out-of-scope vs dropped-as-bad-process separately in the pilot report.

### 5.2 Phase 4 — Consistency filter (the decisive one)

> **Design provenance**: this two-step filter adopts VideoTemp-o3’s data-curation
> verification (their Fig. 2), which is peer-validated: Step-1 “answer using only the
> segment → keep only if correct” ensures the evidence is sufficient; Step-2 closed-loop
> consistency re-check ensures the answer is stable. We reuse that structure BUT replace
> their teacher (Qwen3-VL-235B-Thinking + Gemini-2.5-Pro) with our **own 4B self-backbone**,
> to honor the no-stronger-teacher constraint (§0.1, constraint #1). The point of their
> filter is “is the localized evidence sufficient”; the point of ours is additionally “is
> the reasoning reachable by the target 4B in a single forward pass with no tools”.

For each rewritten CoT, test whether the **target 4B**, given **only the shown frames and
the question (NO tools, vision-only per §1)**, can reproduce the correct answer when
conditioned on (or guided by) that CoT. Procedure (mirrors VideoTemp-o3 Step-1/Step-2 with
self-backbone):

1. **(≈ their Step-1, sufficiency)** Feed `(shown_frames, question)` to the served 4B with
   the rewritten CoT as a forced/seed rationale; check final answer correctness.
1. **(≈ their Step-2, consistency)** Re-run with the CoT as prior context and confirm the
   answer stays correct/stable; also run free-form (no CoT) to record the base gap.
1. Reuse `verify_grounding` as the marker/claim checker on the produced answer.
1. **Keep only CoTs the 4B can act on to reach the correct answer.** Drop the rest — these
   contained steps the 4B cannot execute (residual type-2 leakage). This step is what
   guarantees every surviving CoT is “reachable” by the target model rather than parroted.

### 5.3 De-duplication

Reuse the `PredictionCache` fingerprint idea (case id + model + prompt fingerprint + frames

- code version) to drop near-duplicate trajectories so SFT does not overfit repeats.

### 5.4 Pilot retention gate (IMPLEMENT AS CLI)

`python -m app.distill.pilot --n 50` runs Phase 0→4 on 50 cases and prints:

- Phase 2 strict-pass rate (of generated trajectories).
- Phase 4 consistency-retention rate (of rewritten CoTs).
  Decision rule baked into output text: retention <30% → STOP/Shrink; 30–50% → caution;
  ≥50% → proceed to scale. Do not proceed to Phase 5 automatically.

-----

## 6. Phase 5 — SFT (LoRA)

- **Base**: `Qwen3-VL-4B-Instruct` (NOT Thinking — see §8 rationale: Instruct gives clean
  causal attribution because the base has near-zero spontaneous CoT, so any reasoning the
  probes detect is attributable to our internalization, not to pre-existing RL’d reasoning).
- **Data**: surviving `(frames, question, CoT, answer)` from §5. Target **1k–3k** clean
  samples. Do NOT exceed ~5k (overfitting risk that would corrupt the OOD experiments in §7).
- **Trainer**: LoRA via the author’s preferred stack (Unsloth or 2U1/Qwen-VL-Series-Finetune
  — both support Qwen3-VL). Author handles GPU specifics (4×RTX3080-20GB, no NVLink).
- **Vision-module caveat** (surface this, do not silently violate): if LoRA must touch vision
  modules, do NOT combine 4-bit quant with vision training; use 16-bit for vision params, and
  set vision LR ~5–10× smaller than the language LR.
- **Pre-SFT probe (required)**: before training, run base `Qwen3-VL-4B-Instruct` zero-shot on
  the eval set and record whether it spontaneously emits CoT. Log this baseline; if it does
  emit CoT, add a prompt-template suppression and record it, so the §7 causal probes are not
  confounded by residual base reasoning.

**Output**: a LoRA adapter + a deterministic inference wrapper that produces
`(cot, final_answer)` separably (the probes in §7 need to intervene on the CoT span).

-----

## 7. Phase 7 — Evaluation: Reasoning vs Memorization (design for clean attribution)

This is the scientific core. Implement all probes against a **held-out eval set with strict
video/scene isolation from SFT data**. Eval set sizing: see §7.5 — the legacy 20-case set is
a pilot sanity check ONLY, not for statistical claims.

### 7.1 Distribution-shift battery (Experiment 1)

Three OOD tiers; report IID→OOD *decay slope*, not absolute accuracy:

- **L1 surface shift**: same reasoning structure, different visual domain (e.g. animation /
  low-light / different camera).
- **L2 compositional shift**: reasoning sub-steps recombined in orders unseen in training.
- **L3 depth extrapolation**: train on 2–3 hop chains, test 4–5 hop.

**Mandatory control arm — shuffled-CoT**: train an identical LoRA on the SAME (frames,
question, answer) but with **shuffled/irrelevant CoT**. If our model’s OOD decay curve
matches the shuffled control, the model learned a template, not reasoning. This control is
the strongest defense against reviewer skepticism; it is not optional.

### 7.2 CoT causal intervention (Experiment 2 — most decisive)

- **2a Counterfactual injection**: corrupt one key intermediate conclusion in the model’s own
  CoT (e.g. “there are 3 people” → “5 people”), force continuation, measure **answer flip
  rate**. Real reasoning flips with corruption; template does not.
- **2b CoT ablation**: compare (a) full CoT, (b) no CoT, (c) length-matched irrelevant CoT.
  Real reasoning: (a) ≫ (b) ≈ (c).
- **2c Early-stopping probe**: truncate CoT at varying lengths; real reasoning shows monotone
  accuracy climb, template shows flat/step.

Implement intervention hooks at the CoT/answer boundary (§6 separability requirement).

### 7.3 Visual-dependency probes (Experiment 3 — VLM-specific differentiator)

- **3a image perturbation/swap**: change image so the correct answer should change; does the
  answer follow? Template / language-prior models do not move.
- **3b counterfactual images vs co-occurrence prior**: construct frames that violate training
  co-occurrence (objects normally seen together, deliberately separated) — borrow the
  Common-O adversarial logic. Tests whether the model looks vs. guesses from priors.
- **3c attention/grounding check**: where the CoT claims “looking at t=180s”, verify (via the
  NExT-GQA timestamp GT) that attention/evidence actually lands there. Claimed-but-absent =
  template phrasing.

### 7.4 Data-diagnostic curves (Experiment 4 — cheap, run early)

- Train with vs. without near-duplicate removal; if dedup tanks OOD, memorization dominates.
- Plot OOD accuracy vs. training *diversity* (not count).

### 7.5 Sample-size protocol (compute, do not guess)

- The 20-case set is for **direction sanity only**. For statistical claims on flip rate
  (2a), run a power analysis: a large effect (e.g. 60% vs 15% flip) needs ~15–20 per arm,
  but a moderate effect (45% vs 25%) needs ~80–100 per arm. With multiple OOD tiers ×
  intervention types, budget **≥100 cases per cell, ~300–500 total**, video-isolated.
- Implement `python -m app.distill.power_analysis` that takes a pilot flip-rate estimate
  and outputs required N per arm. **Order of operations**: pilot on 20 → estimate effect →
  compute N → only then label the full eval set. Do not label 500 blindly.

-----

## 8. Phase 6 — GRPO/DAPO (after SFT)

SFT alone is behavior cloning; its ceiling is the (self-generated) trajectory quality and it
cannot self-correct off-distribution. RL with verifiable reward is what turns “internalized
CoT” into robust reasoning and directly attacks the 2a failure mode (hallucinated mid-steps).

- **Base for RL**: the SFT’d 4B (cold-start). 4B chosen over 8B deliberately — on this
  hardware 4B+RL is tractable and prior chart-reasoning results show 4B+RL surpassing 8B
  baselines, which is exactly the “method over parameters” story.
- **Prompts**: 500–2k diverse cases with verifiable reward. No CoT labels needed.
- **Reward** (reuse existing infra as verifiers):
  - rule-based, every sample: final-answer correctness + grounding-marker validity
    (`verify_grounding`).
  - **Penalty-aware temporal reward (adopted from VideoTemp-o3, their Eq. 5–6).** Our model
    does NOT emit `<tool_call>` intervals at inference (no tools), but its CoT cites
    `[FRAME:t=]` anchors (“looking at t=180s …”). Treat the set/range of cited frame
    timestamps as the model’s *implicit* temporal claim and score it against the NExT-GQA
    grounding GT with IoU. Apply their anti-hacking penalty: when `IoU < σ`, subtract `λ`
    from the reward; else use raw IoU. **Starting hyper-params from their paper: σ = 0.1,
    λ = 0.1.** Rationale to preserve: naive IoU reward is hackable (model cites arbitrary
    timestamps to inflate IoU); the penalty term suppresses that. This reward is only
    defined for cases that HAVE grounding GT (NExT-GQA); for grounding-less supplements,
    fall back to answer + marker-validity only.
  - optional periodic judge (existing `JUDGE_*`) every K steps for open-ended.
- **Algorithm**: start GRPO; DAPO if clip-instability appears (author has prior DAPO work).
  GRPO config reference from VideoTemp-o3: group size G = 8, on-policy, group-normalized
  advantage shared across trajectory tokens (good starting point; author tunes for hardware).
- **Anti-forgetting**: prefer orthogonal-LoRA-style constraint (O-LoRA) so RL/SFT does not
  wipe base general ability; document the BWT/forgetting check.
- Keep RL on 4B to fit the no-NVLink 20GB×4 constraint; use a memory-efficient RL stack.

**Post-RL**: re-run the full §7 probe battery. The key comparison: does RL raise the 2a flip
rate (CoT becomes more causally load-bearing) relative to SFT-only?

-----

## 9. Optional Phase — Internalized-vs-Native mechanism study (publication-grade, do last)

Run the §7.2a counterfactual probe on BOTH our internalized-Instruct model AND a native
`Qwen3-VL-4B-Thinking`. If internalized flip-rate ≈ native flip-rate, strong evidence that we
internalized reasoning of comparable causal quality (not a template). If they differ, that
difference is itself a finding about internalized vs. RL-native reasoning. Thinking is used
ONLY here as a comparison ceiling — never as the SFT base (it would confound §7).

-----

## 10. Deliverables & Module Layout (new code only)

```
app/
  vqa.py                      # EXTEND: VLMBackbone protocol + LocalVLMBackbone
  distill/
    __init__.py
    generate.py               # Phase 1: run agent loop, persist full trajectories
    rewrite.py                # Phase 3: trajectory -> CoT, type-1/2 split, validation
    filter_strict.py          # Phase 2: harness strict mode wrapper
    filter_consistency.py     # Phase 4: replay CoT on 4B, keep reachable
    dedup.py                  # Phase 5 prep: fingerprint de-dup
    pilot.py                  # §5.4 pilot retention gate (CLI)
    power_analysis.py         # §7.5 sample-size calculator (CLI)
    build_sft_dataset.py      # emit (frames, question, cot, answer) jsonl
    build_rl_prompts.py       # emit RL prompt set with verifiers
    download.py               # §14: ModelScope-first artifact resolver (models + datasets)
  eval_distill/
    probes_distshift.py       # Exp 1 + shuffled-CoT control
    probes_causal.py          # Exp 2 a/b/c, CoT/answer intervention hooks
    probes_visual.py          # Exp 3 a/b/c
    probes_data.py            # Exp 4
scripts/
  check_local_vlm.py          # Phase 0 endpoint smoke test
data/distill/
  trajectories/  cot/  sft/  rl/  eval/
```

**Out of scope for Codex**: vLLM/SGLang serving config, GPU/DeepSpeed tuning, dataset
licensing. Author owns these.

-----

## 11. Implementation Order (strict; each step gated)

1. Phase 0 (`vqa.py` backbone + `check_local_vlm.py`) — verify existing tests still pass.
1. Phase 1 generate + Phase 2 strict filter on **50 cases**.
1. Phase 3 rewrite + Phase 4 consistency filter on those 50 → run `distill.pilot`.
1. **GATE**: inspect retention. <30% stop & report; ≥50% continue. (caution band 30–50%: author decides.)
1. Scale Phases 1–4 to full; `build_sft_dataset`.
1. Phase 5 SFT + pre-SFT probe; produce CoT/answer-separable inference wrapper.
1. Phase 7 probes on SFT-only model (run `power_analysis` first to size eval set).
1. Phase 6 GRPO/DAPO; re-run Phase 7.
1. Optional Phase 9 mechanism study.

At every gate, print metrics and STOP for human review. Never auto-advance past a gate.

-----

## 12. Non-negotiable invariants (assert in code where possible)

- The trajectory-generating agent’s VLM backbone == the SFT base family (no stronger VLM teacher).
- CoT contains zero tool names, retrieval verbs, or non-visual numeric scores (regex-asserted).
- Every CoT marker resolves to a shown frame (grounding-asserted).
- SFT and eval sets are video-isolated (assert no shared `video_id`).
- Soft-Waive is OFF for training-data selection, ON only for the product eval harness.
- Eval statistical claims require N from `power_analysis`, not the 20-case pilot.
- Training-source videos and eval videos share NO `video_id` (assert; see §13.4).
- **At inference, the model emits NO `<tool_call>` and NO external Crop runs** — single
  forward pass only. If any tool fires at eval time, the run is invalid (this is the entire
  distinction from VideoTemp-o3; see §0.1).
- **VideoTemp-Bench 0–3min split is NEVER used for eval** (NExT-GQA-derived → leaked).
- Do not describe the contribution as “single model” anywhere (code comments, docs, paper) —
  the novelty is “no tool at inference” + “causal verification”, not model count (§0.1).

-----

## 13. Data Acquisition (vision-only)

We do NOT download ready-made CoT datasets. Training data is PRODUCED by running the existing
agent over QA pairs whose videos we have ingested, then filtering (§5). This section defines
WHICH QA sources to use and how to prepare them. The selection criteria are stricter than the
author’s previous orchestrator-distillation runs because the goal changed from “reproduce
agent behavior” to “produce verifiable, internalizable, causally-probeable CoT”.

### 13.1 Source selection criteria (all four required)

1. **Vision-only answerable** — answer obtainable from frames alone; no transcript/OCR/audio.
1. **Temporal grounding GT (strongly preferred)** — enables the “answer-correct AND
   grounding-hit” double filter (§5.1) and the 3c attention probe (§7.3) and RL reward (§8).
1. **Auto-verifiable** — MCQ or short answer with rule-based correctness.
1. **Multi-step visual reasoning** — temporal / counting / comparison / detail; NOT
   single-frame recognition (nothing to internalize) and NOT global “overview”.

### 13.2 Verdict on candidate sources

|Source                         |Vision-only                                                                     |Grounding GT                                |Auto-verify|Decision                                                        |
|-------------------------------|--------------------------------------------------------------------------------|--------------------------------------------|-----------|----------------------------------------------------------------|
|**NExT-GQA**                   |yes (descriptive Qs already removed upstream; test = causal+temporal)           |**YES — per-question timestamp GT, IoP/IoU**|yes (MCQ)  |**PRIMARY training source**                                     |
|Video-MME                      |partial (many Qs need subtitle/audio)                                           |no                                          |yes (MCQ)  |**Supplement — vision-only subset only**                        |
|NExT-QA (parent)               |yes                                                                             |partial                                     |yes        |volume backup if NExT-GQA short                                 |
|Perception Test                |yes (physics/causal)                                                            |partial                                     |yes        |diversity supplement / OOD eval                                 |
|**VideoTemp-Bench** (Kwai-Keye)|mixed (has OCR/synopsis tasks → filter)                                         |yes (their bench has temporal GT)           |yes        |**OOD EVAL ONLY, >3min splits only — see §13.4 leakage warning**|
|**WorldSense**                 |**NO — by design requires audio+video coupling; single modality drops ~15% acc**|no                                          |yes        |**DO NOT USE (out of scope)**                                   |

Rationale Codex must respect: the author previously used Video-MME and WorldSense for
orchestrator distillation. Under vision-only, **WorldSense is dropped entirely** (its tasks
are constructed so neither modality alone suffices), and **Video-MME is demoted to a filtered
supplement** (no grounding GT; must strip subtitle/audio-dependent questions). NExT-GQA
becomes PRIMARY precisely because it is the only source satisfying criteria 1+2+3 together.

### 13.3 Preparation steps (wire into Phase 1)

1. **NExT-GQA (primary).** Reuse `scripts/eval_convert_nextgqa.py`, but AUDIT it first: it must
   **retain the per-question temporal grounding GT** (start/end timestamps). If the existing
   converter dropped this field (it may have, since old eval didn’t need it), patch it to keep
   it. Persist GT into the case record so §5.1 and §7.3c can read it. Note NExT-GQA scores
   grounding with **IoP ≥ 0.5**; use that threshold for the “grounding-hit” gate.
1. **Question-type pre-screen (save compute before running the agent).** Keep only
   `temporal_order`, `counting`, `comparison`, `visual_detail`, `existence`. Drop `overview`
   and trivially single-frame recognition. NExT-GQA’s test split is already ~57% causal /
   ~43% temporal with descriptive removed, so most of it qualifies.
1. **Video-MME (supplement).** Filter to a vision-only subset: drop any question whose stem or
   reference reasoning references narration/subtitle/audio (“according to the speaker”, etc.).
   No grounding GT, so these cases pass §5.1 on answer-correctness only (flag them so the
   grounding-dependent probes skip them).
1. **Ingest** the source videos through the existing `preprocess` pipeline so the agent has
   its evidence indices before Phase 1 runs.

### 13.4 Train/eval isolation (decide BEFORE producing data —泄漏会让 §7 结论作废)

NExT-GQA is also the author’s CURRENT product benchmark (the 70%/90% numbers). Using it as
BOTH training source and eval set leaks. **Adopt option 2:**

- **Training source = NExT-GQA (full)**, used to produce CoT + feed grounding-based filters/RL.
- **Eval main axis = a vision-only set with NO video overlap** — e.g. Perception Test and/or
  the Video-MME vision-only subset, held out entirely from training.
- This doubles as the §7.1 **L1 distribution-shift** test (cross-dataset eval is naturally OOD),
  satisfying data-isolation and OOD evaluation in one move.
- Assert at dataset-build time: `set(train_video_ids) ∩ set(eval_video_ids) == ∅`.

**VideoTemp-Bench usage — leakage hazard, read carefully.** VideoTemp-Bench (the newest
long-video GQA benchmark, from the VideoTemp-o3 authors) is attractive as an OOD axis because
it stratifies by duration (0–3 / 3–10 / 10–20 / >20 min) — a ready-made dimension for showing
how internalized single-forward reasoning degrades with video length. BUT:

1. **Its 0–3min split is drawn from NExT-GQA** (their Appendix A). Since we train on NExT-GQA,
   the 0–3min split is LEAKED — **must be excluded from eval**.
1. Therefore use **ONLY the >3min splits** (3–10 / 10–20 / >20 min; sourced from LongVILA /
   LongVideo-Reason / ScaleLong), and still run a video_id leakage check against the training set.
1. It mixes audio-ish / OCR / information-synopsis tasks → apply the same vision-only
   case-scope filter (§1) before using it for eval.
1. Use it as **OOD evaluation only, never as a training source.**

- Assert: VideoTemp-Bench eval cases share no `video_id` with training, AND none come from its
  0–3min (NExT-GQA-derived) split.

### 13.5 Volume expectation

NExT-GQA test ≈ 5,553 questions / 990 videos. After question-type pre-screen + the three
filters (§5), assuming ~20–30% net retention, this yields roughly the SFT target of 1k–3k
clean CoTs. If short, expand with the NExT-QA parent set (same videos family — re-check
isolation) before touching lower-quality sources.

-----

## 14. Download-source policy (ModelScope first)

The author is in mainland China; HuggingFace is slow/unreliable. **Prefer ModelScope for ALL
model and dataset downloads; fall back to HF only when ModelScope lacks the artifact.**

- **Models** (e.g. `Qwen3-VL-4B-Instruct`, BGE-M3, SigLIP2): pull from ModelScope.
  Use `modelscope download` CLI or `from modelscope import snapshot_download`. Set
  `MODELSCOPE_CACHE` to a project-local dir.
- **Datasets**: try ModelScope dataset hub first (`from modelscope.msdatasets import MsDataset`
  or `modelscope download --dataset <id>`). NExT-GQA / NExT-QA / Video-MME / Perception Test
  ModelScope mirrors are NOT guaranteed to exist or to be complete (NExT-GQA’s grounding
  annotations especially may be missing on mirrors). **Resolution order per artifact:**
1. ModelScope mirror, IF it contains the grounding GT / required splits intact.
1. else the original release (NExT-GQA: `github.com/doc-doc/NExT-GQA`; raw videos from VidOR)
   or the HF dataset, with an env-var HF endpoint mirror (`HF_ENDPOINT=https://hf-mirror.com`).
- **Implement a single resolver** `app/distill/download.py` with `fetch(artifact_id, kind)` that
  tries ModelScope then falls back, logs which source served each artifact, and verifies the
  grounding-GT field is present for NExT-GQA before accepting a mirror. Do NOT silently accept a
  mirror that is missing grounding annotations — that would quietly break §5.1/§7.3c.
- Never hardcode HF URLs in pipeline code; route everything through the resolver. The author
  handles network/proxy at the shell level; you just honor the ModelScope-first order.

-----

## 15. Implementation deltas / 设计修订 (2026-06-08)

A design review found that the original §5.4 pilot gate would **pass for the wrong
reasons**: the root cause is that *where the frames come from at inference* was never
defined. The thesis was narrowed accordingly and 7 changes were implemented. Sections
§0–§14 above are left intact for audit; **this section overrides them where they conflict.**

**Narrowed thesis.** From "internalize the whole agent's multi-step reasoning" to:
*when the relevant frames are visible (short-video regime), tool-scaffolded multi-step
visual reasoning can be compressed into a single tool-free forward pass, and causal probes
show it is genuine reasoning, with a characterized internalizability boundary.* The two
non-negotiable selling points are unchanged (single forward / no iterative tool; causal
verification). **Frame *selection* is now explicitly declared non-internalizable and out of
scope** — this pushes the type-1/type-2 boundary down from "tool name" to "capability".
NExT-GQA's short videos (~40s @ 1fps) make a fixed uniform sample very likely to contain the
evidence, which is exactly why the frame-selection confound is small in this regime.

| # | Spec location | What the spec says | What is now implemented |
|---|---|---|---|
| 1 | §0.1, §4.1 frame source | "single forward pass from only the frames it can see" — frame origin undefined; trajectory captures agent-retrieved `shown_frames` | **Fixed, query-agnostic uniform 16-frame sampler** supplies frames at generation/CoT/consistency/SFT/inference. A one-shot preprocessing function, NOT a per-query tool loop, so "no tool at inference" stays honest; identical across base/internalized so frame selection is held constant. Code: `app/distill/frames.py`, config `distill_sampler_frames=16`. |
| 2 | §2 pipeline | Phase 1 records agent-shown frames | The uniform sample (not `retrieved_frames`) is the frame set carried through `state.sampler_frames`; SFT trains on the same frames the model sees at eval (removes train/inference mismatch). |
| 3 | §5.1 Phase 2 | text/audio case-scope filter only | **Added evidence-coverage filter**: drop a case when the uniform sample misses the GT window (`evidence_not_in_uniform_sample`); GT-less supplements skip the check and are flagged `no_grounding_gt`. |
| 4 | §5.2 Phase 4 | "keep CoTs the 4B can act on to reach the correct answer" | Gate now tests *reachability*, not answer-copying: the seed CoT has its final answer **stripped**, and the keep criterion is **`conditioned_ok AND consistency_ok AND NOT free_ok`** (the base must not already get it free-form). New headline metric `signal_gain_rate` (will read **lower** than the old retention — that is the honest signal). |
| 5 | §13.4 isolation | train on NExT-GQA **full** | **Video-disjoint train/held-out split** added so §7.1 has a clean IID point; held-out NExT-GQA = IID, cross-dataset = L1 OOD. Code: `scripts/split_cases.py`. |
| 6 | §8 RL reward | mandatory penalty-aware temporal-IoU (σ=0.1, λ=0.1) | **Disabled deliberately.** Under fixed-frame / no-tool, the cited `[FRAME:t=]` anchors are set by the sampler, not a model action, so the implicit-claim IoU reward is not load-bearing and is hackable. Reward = answer-correctness + marker-validity only; GT is still carried for probe 3c. |
| 7 | §7.3c probe | "attention/grounding check" | **Renamed to claimed-timestamp-vs-GT consistency.** OpenAI-compatible serving exposes no attention maps; we instead verify the CoT's cited `[FRAME:t=]` markers land on the grounding GT. Code: `app/eval_distill/probes_visual.py --check-claims`. |

**Incidental fix.** Grounding GT (`gold_timestamps`/`gold_scenes`) now propagates from the
trajectory through the CoT artifact into the SFT/RL rows (§3 GT-passthrough bug; it was
previously dropped at the rewrite stage), restoring the inputs §7.3c needs.

**Also added (§7.1 mandatory control).** Shuffled-CoT control dataset builder:
`app/eval_distill/probes_distshift.py --build-shuffle-control` derives the control arm
(same frames/question/answer, mismatched CoT) from the SFT JSONL.

**Feasibility verdict.** Even if the corrected `signal_gain` set is small — 4B single-forward
may already cover much type-1 visual reasoning on short video — that is itself a publishable
**internalizability-boundary** result (a stated selling point), so the project remains worth
running; the 50-case gate now measures the right quantity and returns an honest go/no-go in
1–2 days.

### Updated invariants (amend §12)
- Frames at inference come ONLY from the fixed uniform sampler; no retrieval/crop tool runs.
- SFT/RL/consistency frame sets == the uniform sample (assert: not `retrieved_frames`).
- Phase-4 keep requires `NOT free_ok` (no base-already-correct samples in the kept set).
- Temporal-IoU reward stays OFF unless the model emits an explicit frame-selection action.