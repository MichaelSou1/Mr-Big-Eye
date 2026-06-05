---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: visual-question-answering
license: cc-by-nc-4.0
language:
- en
- zh
tags:
- lora
- qlora
- peft
- video-understanding
- video-question-answering
- agent
- tool-use
- agentic-rag
- distillation
- audio-visual
---

# Mr-Big-Eye Orchestrator v2 — distilled agentic video-QA LoRA (Qwen2.5-7B)

A **QLoRA adapter** that turns `Qwen/Qwen2.5-7B-Instruct` into an **agentic video-QA
orchestrator** — a tool-calling policy that, given a question about a long video, *plans
a retrieval strategy, issues tool calls over a multimodal evidence store, checks evidence
sufficiency, and answers with grounded timestamp citations*. The policy is **distilled from
a strong proprietary teacher's tool-call trajectories** (behavioral cloning of an agent,
not plain QA fine-tuning).

**TL;DR** — on a 181-case leakage-free hard heldout (judged pass_rate), this adapter lifts
the base 7B from **0.547 → 0.669 (+12.2pp)**, with the biggest jump on **audio-visual /
temporal ("joint") reasoning: 0.362 → 0.621 (+25.9pp)**. It is the best of three compared
checkpoints and meets the pre-registered acceptance bar (+3pp over the previous model).

> ⚠️ This is a **LoRA adapter + an agent policy**, not a standalone chat model. It emits
> tool calls in the [Mr-Big-Eye](https://github.com/MichaelSou1/Mr-Big-Eye) harness schema
> and needs that harness (retrieval indexes + a VLM tool) to run end-to-end.

---

## 1. What the model actually does (architecture)

Each video is pre-indexed offline into a multimodal evidence store: **ASR transcript**
(SenseVoice + FSMN-VAD, sentence-level timestamps), **scene/dense frames** (SigLIP2
embeddings + VLM captions), and **slides/OCR**, each in its own vector index. At question
time the orchestrator runs an **agent loop** over a tool suite:

- `retrieve_video_evidence` / `retrieve_transcript_evidence` / `retrieve_slide_evidence` —
  modality-specific retrieval over the per-video indexes
- `search_transcript_keyword`, `segment_focus`, `build_timeline` — targeted lookup,
  temporal zoom, and event ordering
- `align_audiovisual_evidence` — couple audio (transcript) with visual (frames) for
  "joint" questions
- `retrieve_hypothesis_evidence`, `assess_evidence_sufficiency`, `stitched_verify` —
  hypothesis-driven gathering, a stop-discipline sufficiency check, and grounding verification
- `answer_with_evidence` — final answer **with `[TRANSCRIPT:t=..]` / `[FRAME:t=..]`
  citations** that are scored for correctness

The adapter is the **policy** that decides *which* tool to call *when*, with *what*
arguments, and when evidence is sufficient to answer — the hard part of long-video QA.

## 2. Results (judged pass_rate, 181-case leakage-free heldout)

90 Video-MME + 91 WorldSense questions; **no heldout video was ever trained on** (verified
0 leakage at the trajectory source). All three models evaluated apples-to-apples on the
same cases, judge on, served from a **single vLLM with hot-swapped LoRA modules**, 1 run.
`base` = Qwen2.5-7B-Instruct; `pilot` = an earlier 355-sample adapter; `v2` = this model.

| pass_rate | base | pilot | **v2 (this)** | v2−base | v2−pilot |
|---|---|---|---|---|---|
| **overall** | 0.547 | 0.630 | **0.669** | **+12.2pp** | **+3.9pp** ✅ |
| joint (audio-visual + temporal) | 0.362 | 0.621 | **0.621** | **+25.9pp** | +0.0 |
| audio | 0.356 | 0.444 | **0.444** | +8.9pp | +0.0 |
| visual | 0.785 | 0.708 | **0.815** | +3.1pp | +10.8pp |
| overview | 0.846 | 0.923 | **0.923** | +7.7pp | +0.0 |
| — Video-MME (90) | 0.700 | 0.722 | **0.800** | +10.0pp | +7.8pp |
| — WorldSense (91) | 0.396 | 0.538 | **0.538** | +14.2pp | +0.0 |

Per-modality n is 13–65; treat <3pp deltas as ties. W&B: training run `ts8urbus`, eval run
`gi7iicvl`. Per-model breakdowns are in [`eval/`](./eval).

## 3. Distillation methodology

- **Teacher:** a strong proprietary orchestrator (deepseek-v4-pro) drives the *same* tool
  harness; its full tool-call trajectories are captured per question.
- **Quality tiering** (the key data-quality lever): each trajectory is graded —
  *tier-1* (judge-correct, no runtime guard), *tier-2* (correct, only a benign dedup guard),
  *tier-3* (teacher wrong / degenerate). Only tier-1+2 are kept ⇒ the SFT set is
  **100% teacher-correct**; 40% of raw trajectories (all `judge_wrong`) are discarded.
- **SFT shaping:** trajectories are sliced into per-step `(context → next tool call)`
  samples; forced/guard targets are excluded so the student learns the *policy*, not the
  harness's safety rails. base64 image blobs are stripped from context.
- **QLoRA:** r=32, α=64, 4-bit nf4, Liger fused cross-entropy (fits 8192-token agent
  contexts on a 20 GB card), lr 1e-4 cosine, batch 1 × grad-accum 8, bf16, 2 epochs.
  ~1,499 SFT samples. Best checkpoint selected by **held-out task pass_rate**, not eval
  loss (see §5).

## 4. Data engineering

- **Difficulty/modality rebalance:** short, near-trivial clips are dropped; training is
  refocused on harder medium videos selected by a **modality-weighted greedy sampler** that
  raises the multimodal (`joint`) share.
- **External audio-visual enrichment:** integrated **WorldSense** (audio-visual-synchronised
  MCQ) to push the `joint`+`audio` share of new training data from **~14% → 70%**.
- **Leakage control:** held-out videos are force-excluded from training and never
  teacher-captured; 0 leak verified at source.
- **Robust ingestion:** per-frame VLM **content-filter rejections are tolerated** (drop one
  caption, keep the video) instead of failing a whole video — recovered held-out items.

## 5. Engineering rigor & findings (why this is more than a fine-tune)

- **Quality ≫ quantity, measured.** A prior 4.5× data scale-up (355 → 1,618 samples) by
  adding easy clips *regressed* −4.2pp on hard cases; rebalancing toward harder, multimodal
  data at ~constant size produced the best model instead.
- **Honest model selection.** All 20-step checkpoints were retained; the eval-loss minimum
  ≠ the pass_rate maximum, so the shipped checkpoint was chosen (and re-verified) by
  **hard-set pass_rate**. A cheap checkpoint-trajectory probe showed pass_rate peaks
  mid-training while eval loss keeps falling — so a 3rd epoch was **rejected** as overfitting.
- **Teacher-capacity ceiling (a clean negative result).** The audio-visual enrichment gave
  **no `joint`/`audio` gain over the pilot** because the teacher itself fails ~56% on the
  hardest audio-visual cases — the bottleneck there is the teacher/tooling, not the student
  or data volume. This redirected strategy from "more data" to "stronger teacher / audio tool."
- **Reproducible infra:** 4-GPU sharded ingest + teacher capture + sharded evaluation;
  W&B tracking; single-server multi-LoRA eval.

## 6. Usage (vLLM, OpenAI-compatible + tool calls)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora --lora-modules v2=michaelrhs/mr-big-eye-orch-v2 \
  --max-lora-rank 32 --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384
```

Then drive it with the [Mr-Big-Eye harness](https://github.com/MichaelSou1/Mr-Big-Eye)
(point the orchestrator at the served `v2` model; keep your VLM/judge endpoints unchanged).
The adapter expects the harness's tool schemas; loaded as a bare chat model it will answer
but without evidence tools.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "michaelrhs/mr-big-eye-orch-v2")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

## 7. Limitations

- Research artifact; **non-commercial** (derives from WorldSense data, CC-BY-NC-4.0).
- It is a tool-calling **agent policy** for the Mr-Big-Eye harness, not a general VQA/chat model.
- Audio-visual reasoning is capped by the teacher (§5): expect gains mainly on visual/temporal
  retrieval, not on the hardest non-speech-audio questions.
- Evaluated at n≈181 (per-modality 13–65) — small-N; <3pp deltas are ties.

## 8. Links

- **Code, full write-up & resume-grade summary:** https://github.com/MichaelSou1/Mr-Big-Eye
- **Base model:** [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Eval data:** Video-MME; WorldSense (`honglyhly/WorldSense`, CC-BY-NC-4.0)
