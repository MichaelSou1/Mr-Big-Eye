# Reading, Not Reasoning: A Causal Audit of Internalizing Agentic Visual Reasoning into Single-Forward VLMs

> **Stale convenience draft.** The canonical upgraded manuscript is
> [`docs/paper/paper.tex`](paper/paper.tex), updated on 2026-06-26 for WU-6/N4.
> This Markdown file still preserves the 2026-06-12 pre-upgrade n=60 narrative and should not be used
> as the submission source without regeneration from the LaTeX.

**Ruihan Su** · Sun Yat-sen University · surh6@mail2.sysu.edu.cn

*Short paper draft — 2026-06-12. All numbers regenerate from `data/distill/results/results.jsonl` via `scripts/{regen_tables,build_map,power_table}.py`; probe artifacts in `data/distill/poc/`.*

## Abstract

A popular recipe for efficient multimodal reasoning is to *internalize* an agent's multi-step trajectory — reflection, orchestration, tool use — into a single tool-free forward pass of a small vision-language model (VLM). We ask two questions this literature rarely separates: **(1) where does agentic reasoning over fixed visual evidence have any headroom to internalize, and (2) when distillation does lift accuracy, is the internalized chain-of-thought (CoT) causally load-bearing?** We build a variance-gated map over 3 datasets (NExT-GQA, CLEVRER, ChartQA), 4 model scales (4B–32B), and **three vision-encoder paradigms** (Qwen3-VL tiling-ViT, InternVL3 fixed-resolution ViT, Penguin-VL's LLM-based encoder). The map is two-regime and stark: in the frames-visible **video** regime, no agentic/reflective method reliably beats a single forward pass in *any* cell — a multi-seed gate retracts our own initially promising +8% as run-variance — and the only statistically reliable agentic effects are **negative** (orchestration corrupts answers by 8–16%). A **reasoning-bound** regime appears only once a strong base solves perception (32B on static charts). In that regime we obtain a genuinely significant internalization gain (8B ChartQA: .617→.733, McNemar p=.023) and a clean 32B QLoRA gain (.700→.767, 4 gained / 0 lost). But a Lanham-style causal probe over a 2×2 of {8B, 32B} × {image present, image masked} shows the internalized CoT is **not load-bearing under real inference**: with the chart visible, corrupting a CoT intermediate flips the answer no more than shuffling the CoT (32B: 2.2% ≈ 2.2%); a latent reasoning chain surfaces only when the image is masked. The stronger base bypasses its CoT *more*, independently reproducing "larger ⇒ less faithful CoT" in a VLM. We conclude that trajectory distillation transfers *perception/reading*, not load-bearing reasoning — and that accuracy-only evaluations of "internalized reasoning" systematically overclaim.

## 1 Introduction

Agentic VLM systems — iterative frame retrieval, self-reflection, critic-orchestrated re-reading, visual tool use — report gains over single-pass inference, motivating a distillation step: convert the trajectory into a CoT and fine-tune a small VLM to reproduce it tool-free at inference (e.g., Visual Program Distillation, PEARL, STAR). Two assumptions are implicit and largely untested:

- **A1 (headroom):** holding visual evidence fixed, the agent's *reasoning* (not its frame selection) beats the base model's single forward pass, so there is something type-1 (single-forward-doable) to internalize.
- **A2 (faithfulness):** when distillation lifts accuracy, the student's emitted CoT is the causal mechanism of the gain.

We test both. For A1 we run a multi-seed, paired-bootstrap **variance gate** over every (dataset × model × agentic-method) cell, with a label audit, a frames-visible partition, and an explicit power analysis. For A2 we run a causal CoT-intermediate probe (corrupt one numeric intermediate vs. shuffle the whole CoT, force-continue) on the SFT'd students, with an **image-masked control** that disambiguates "no reasoning learned" from "reasoning learned but bypassed".

**Contributions.**
1. A statistically-gated **two-regime map**: video QA with the evidence on-screen is perception/selection-bound at every scale (4B/8B/32B) and across three encoder paradigms; agentic nets are within run-variance everywhere, and the only reliable effects are negative. The reasoning-bound regime exists, but only where a strong base has already solved perception (32B + static charts).
2. A cautionary **retraction case study**: our own single-seed "+8% orchestrated reflection" headline washes out to +0.9% with sign flips across 10 seeds — single-seed agentic gains on n≈50 benchmarks are not evidence.
3. Two **positive internalization PoCs** in the reasoning-bound cell (8B: +11.7%, CI excludes 0, McNemar p=.023; 32B QLoRA: +6.7%, 4/0 gain/loss, underpowered p=.125), with per-epoch held-out selection shown to be load-bearing methodology.
4. The first (to our knowledge) **causal CoT-intermediate audit of an internalized, no-tool, single-forward VLM**, with an image-masked control. Verdict: the distilled gain is perception/reading transfer; the CoT is bypassed under real inference, more so at larger scale.

## 2 Related work

**Trajectory/agent distillation.** VPD (Hu et al., CVPR 2024) distills visual-program traces into single-pass VLMs and describes the result as "faithful" on accuracy evidence alone; PEARL (Adhikari & Lapata, arXiv:2604.08065) internalizes expert tool-use trajectories in latent space, avoiding tool invocation at inference; STAR (arXiv:2602.09829) distills multi-agent trajectories (planning, tool use, self-reflection) into one student — all evaluated by accuracy, none causally probing the student's CoT. DualDistill's Agentic-R1 (Du et al., EMNLP 2025), Agent0-VL (arXiv:2511.19900), Pixel-Reasoner, and DeepEyes keep tool invocation at inference and are therefore outside the single-forward setting.

**Causal probing of CoT.** Lanham et al. introduce corrupt/truncate interventions on text-LLM CoT and find larger models produce less faithful CoT. **CodeV** (*Code with Images for Faithful Visual Reasoning via Tool-Aware Policy Optimization*, Hou et al., arXiv:2511.19661, CVPR 2026 Oral) is the closest work: it applies causal perturbations (Mask/Noise/Random/Empty) — but to **tool-output images** of a tool-using agent, i.e., perception inputs, not CoT intermediates of an internalized model. The intersection we occupy — Lanham-style intermediate corruption on a distilled no-tool VLM, plus a perception-vs-reasoning regime boundary and an image-masked control — appears unoccupied as of June 2026.

**Failure taxonomies.** Diagnostic studies report that most VLM "reasoning" failures are perceptual: VRIQ (Khezresmaeilzadeh et al., arXiv:2602.05382) attributes 56% of failures to perception alone, 43% to perception-plus-reasoning, and only 1% to reasoning alone — consistent with our regime-1 mechanism.

## 3 Experimental setup

**Tasks.** *NExT-GQA* (short-video MCQ with grounding GT; 16 uniformly-sampled frames; the **frames-visible regime** — we verify all 50 pilot cases are EVIDENCE_IN, i.e., the uniform sample hits the GT window), *CLEVRER* (collision reasoning, 70 cases), *ChartQA* (static-chart numeric QA, 60 held-out test items). A DeepSeek text-only label audit on NExT yields 47 clean / 3 ambiguous / 0 wrong-gold; all video gates run on the CLEAN∩EVIDENCE_IN n=47.

**Models.** Qwen3-VL 4B / 8B / 30B-A3B (MoE) / 32B (dense), InternVL3-8B (fixed-res ViT, cross-family control), Penguin-VL 2B/8B (LLM-based vision encoder + temporal token compression, cross-*paradigm* control). Local vLLM serving; the orchestrator/critic is a text-only LLM (initially a local Qwen3-30B-A3B, later DeepSeek API), never a stronger VLM teacher for the video gates.

**Agentic methods (frames held fixed = the type-1 isolation).** `self_reflect` (2-turn tool-free self-re-examination over the same frames), `orch_reflect_blind` (text critic poses sub-questions, VLM is the eyes, critic integrates), `orch_reflect_sighted` (32B-VLM critic that can re-read the image). Frame retrieval is deliberately excluded: it changes the evidence (type-2) and cannot be internalized into a fixed-input forward pass.

**Statistics.** K seeds per cell (K=10 for 4B/8B-NExT, 6 for 32B, 3–5 elsewhere), paired bootstrap 95% CIs on the net (gain − loss) over cases, McNemar exact tests for SFT deltas, and a power table: at n=47 the minimum detectable net at 80% power is **≈25%** — so all |net| < 9% nulls are reported as "no reliable effect, underpowered", with cross-seed sign-flips as the power-independent evidence.

## 4 Result I: the two-regime map

| dataset × model | free acc | best agentic net | verdict | regime |
|---|---|---|---|---|
| NExT 4B / 8B / 32B | .74 / .66 / .70 | −.045 / +.060 / +.039 | within-variance (K=10/10/6) | **1** perception/selection-bound |
| NExT InternVL3-8B | .57 | orch **−.163** | **reliable NEGATIVE** (CI [−.32,−.02]) | 1 (cross-family) |
| NExT Penguin-VL 2B / 8B | .53 / .43† | +.050 / +.128 | within-variance (K=3) | 1 (cross-paradigm) |
| CLEVRER 4B–32B | .43–.50 | ≈0 | ≈chance; tracking wall | 1 |
| ChartQA 4B / 8B | .67 / .62 | +.023 / +.012 (4B orch **−.122**, reliable neg.) | within-variance | 1 perception-bound (value reading) |
| ChartQA Penguin 2B / 8B | **.82 / .80** | −.078 / −.006 | within-variance | 1 (perception solved by encoder) |
| **ChartQA 32B** | **.80** | +.017 | within-variance | **2 reasoning-bound** |

†Penguin-8B-NExT ran frame-starved at 2 frames (memory limit); not comparable to 16-frame cells.

**(a) No cell shows reliable agentic headroom; the reliable effects are negative.** Across 13 cells, no agentic/reflective method beats single-forward outside run-variance. Three cells show statistically reliable *harm* from orchestration: 4B-ChartQA −12.2%, InternVL3-NExT −16.3% (all seeds negative), Penguin-2B-ChartQA −7.8%. Per-case dumps show two mechanisms: leading sub-questions make the suggestible VLM emit contradictory new reads (video), and decompose-and-reintegrate corrupts an answer whose sub-reads were correct (charts). Multi-step orchestration adds more failure points than it fixes.

**(b) The retraction.** Our single-seed sweep had one positive headline: 8B-NExT orchestrated reflection +8%. Ten seeds give per-seed nets of +.09, −.04, +.11, +.04, +.02, .00, +.06, −.06, −.06, −.06 — pooled **+0.9%**, sign-flipping. The variance source is the critic's stochastic sub-questions, not the VLM's decoding (self-reflect σ_decode ≈ 0 on MCQ). Single-seed agentic gains at this n are noise.

**(c) Why regime 1 is perceptual: converging probes.** On the 18 8B-NExT free-form errors, giving the model the *ground-truth* frames at maximum resolution recovers only 4/18 (22%); a near-oracle (32B captions of GT frames → text-LLM reasons) recovers 6/18 (33%). Two-thirds of the failure surface is unreachable by any reasoning-over-fixed-frames intervention. The sighted-critic ablation sharpens the mechanism: a 32B sighted critic trends positive on every seed on **charts** (+.070, where the wall is re-readable values) but not on **video** (+.060, CI crossing 0, where the wall is temporal localization that re-reading the same 16 frames cannot fix).

**(d) The encoder-paradigm result.** Penguin-VL's LLM-based encoder lifts small-model chart reading to **.82 (2B)** — matching Qwen-32B (.80) and far above Qwen-8B (.62) — proving the small-model chart-perception wall is an **encoder artifact, not a capability limit**. Yet the temporal-video wall is unmoved (Penguin-2B NExT .53) and agentic nets stay ≤ 0. The regime structure is a property of the *task*, robust across three encoder paradigms.

## 5 Result II: internalization works in regime 2 — by accuracy

In the one reasoning-bound cell, we run the actual distillation. A 32B teacher generates step-by-step chart CoTs on ChartQA-train; 150 rationales pass an answer-consistency filter; LoRA-SFT (vision tower frozen) the student; evaluate single-forward, tool-free on the 60 disjoint test items.

- **8B student:** .617 → **.733**, net **+0.117**, bootstrap CI [+.050, +.200] excludes 0, gain 7 / lost 0, **McNemar p = .023**. A real, significant single-forward internalization gain.
- **32B student (QLoRA, NF4 + r=16 on 4×20GB):** base-with-CoT-prompt .700 → **.767** at the epoch-1/2 peak, gain 4 / lost 0, but McNemar p = .125 (underpowered at n=60). Two methodological notes generalize: (i) train-loss is monotone-meaningless for a 150-example LoRA — test accuracy peaks at epoch 1–2 and *declines* by epoch 4 (.717, losses appear), so **per-epoch held-out selection is load-bearing**; a blind 3-epoch recipe picks a strictly worse checkpoint. (ii) The CoT prompt itself *costs* the untrained 32B ~10% (free-form .80 vs. CoT-prompted .70); the SFT mostly recovers that penalty and does not exceed the base's best free-form .80 — the clean ±adapter contrast is the correct causal isolation, but headline "+X% from CoT distillation" numbers that compare against a CoT-prompted base deserve scrutiny everywhere.

## 6 Result III: the causal probe — the gain is reading, not reasoning

For every test case the student answers correctly with an emitted CoT, we intervene Lanham-style: **corrupt** one numeric intermediate (a targeted hit on the arithmetic chain) vs. **shuffle** the CoT sentences (a control that destroys prose coherence but no specific value), then force-continue to a final answer. If the chain is load-bearing, corrupt-flips ≫ shuffle-flips. We run the full 2×2 of {8B, 32B} × {image present = real inference, image masked = CoT is the only information}, in one in-process harness:

| base | condition | n | corrupt-flip | shuffle-flip | reading |
|---|---|---|---|---|---|
| 8B | image present | 48 | 16.7% | **27.1%** | shuffle ≥ corrupt → chain not load-bearing |
| 8B | image masked | 48 | **37.5%** | 27.1% | corrupt > shuffle → latent chain exists |
| 32B | image present | 46 | 2.2% | 2.2% | CoT ignored entirely; model re-reads chart |
| 32B | image masked | 46 | **10.9%** | 6.5% | corrupt > shuffle → latent chain exists |

Three findings:

1. **Under real inference (image present), the internalized CoT is not load-bearing at either scale.** The 8B is more disturbed by scrambled prose than by a falsified number; the 32B is disturbed by neither (≈2%) — it silently re-derives the answer from the chart. The accuracy gains of §5 are **perception/reading transfer**: the teacher's trajectory taught the student *where and how to read*, not an arithmetic chain the answer rides on.
2. **The stronger the base, the more it bypasses its own CoT** (corrupt-flip 2.2% vs. 16.7%). This independently reproduces Lanham et al.'s "larger models produce less faithful CoT" in a VLM, and identifies the mechanism: a perceptual re-read shortcut that a stronger vision tower can lean on.
3. **The image-masked control is essential.** Masking flips both scales to corrupt > shuffle: a usable latent chain *was* internalized, but is only consulted when the perceptual shortcut is removed. Without this control, the present-condition ≈0% would be misread as "no reasoning learned"; the truth is "reasoning learned but bypassed". This also disposes of the objection that our corruption is too weak to detect any chain.

## 7 Discussion and limitations

**Implications.** (i) Claims of "internalizing agentic reasoning" need a causal audit — accuracy deltas, even significant ones, are compatible with pure perception transfer, and in our engineered best-case regime-2 cell that is exactly what they were. (ii) The agent's real edge on video is *perception routing* (frame/region selection) — which is type-2 by construction and cannot be internalized into a fixed-input forward pass; text-level reflection over a fixed read cannot repair perceptual error and reliably amplifies it. (iii) Single-seed evaluation of agentic methods on n≈50 benchmarks is uninterpretable; report seeds and paired CIs.

**Limitations.** Sample sizes are small (n=47 video / n=60 charts; MDE ≈25% on video at 80% power) — regime-1 nulls are "no reliable effect, underpowered", with cross-seed sign-flips as the power-independent evidence; scaling n is the first next step. The 32B student is QLoRA on an NF4 base (a bf16 full-precision replication is hardware-deferred). The teacher pool is one family (Qwen3-VL); the cross-family/paradigm controls cover students, not teachers. CoT corruption targets numeric intermediates; richer interventions (truncation, paraphrase, step deletion) would tighten the audit. Penguin-8B's video cell is frame-starved and not comparable.

## 8 Conclusion

We set out to compress tool-scaffolded visual reasoning into one forward pass and to verify it causally. The defensible result is a boundary: **where the frames are already visible, there is nothing reliable to internalize — the bottleneck is perception, and agentic reflection is at best noise and at worst reliably destructive.** Where a strong base clears perception, distillation lifts accuracy significantly — but the causal probe shows the student learned to *read* like its teacher, not to *reason* like it, and the stronger the student, the more thoroughly it ignores the chain it was taught to emit. Reasoning that survives a causal audit remains, on this evidence, un-internalized.

---

### Reproducibility

Envs: `mbe-ingest` (gates/SFT/eval), `vllm-qwen` (serving), `vlm_dapo` (QLoRA/probe), `penguin` (Penguin-VL, transformers 4.51.3). Gates: `scripts/run_{variance,chartqa}_gate.py --seeds K --concurrency 8`; floor/partition/power: `scripts/{regrade_dumps,label_audit,build_partition,power_table}.py`; probes: `scripts/diag_{perception_headroom,oracle_perception}.py`; SFT PoC: `scripts/poc_{gen_cot,sft,merge,eval}.py`, `scripts/poc_sft_32b_qlora.py`; causal probe: `scripts/poc_causal_probe_32b.py --base {8B|32B} --adapter <dir> [--mask-image]`. Result store: `data/distill/results/results.jsonl` (fingerprinted, append-only).

### References (key, web-verified 2026-06-12; full BibTeX in [paper/references.bib](paper/references.bib))

- Lanham et al., *Measuring Faithfulness in Chain-of-Thought Reasoning*, arXiv:2307.13702.
- Hou et al., *CodeV: Code with Images for Faithful Visual Reasoning via Tool-Aware Policy Optimization*, arXiv:2511.19661 (CVPR 2026 Oral) — causal perturbation of tool-output images in a tool-using VLM agent.
- Hu et al., *Visual Program Distillation* (VPD), CVPR 2024.
- Adhikari & Lapata, *Multimodal Latent Reasoning via Predictive Embeddings* (PEARL), arXiv:2604.08065.
- Du et al., *Agentic-R1: Distilled Dual-Strategy Reasoning* (the DualDistill framework), EMNLP 2025 — student keeps tools at inference.
- *Internalizing Multi-Agent Reasoning for Accurate and Efficient LLM-based Recommendation* (STAR), arXiv:2602.09829.
- Liu et al., *Agent0-VL*, arXiv:2511.19900; Su et al., *Pixel Reasoner*, arXiv:2505.15966; Zheng et al., *DeepEyes*, arXiv:2505.14362 (tool-at-inference agents).
- Khezresmaeilzadeh et al., *VRIQ: Benchmarking and Analyzing Visual-Reasoning IQ of VLMs*, arXiv:2602.05382 (failure taxonomy: 56% perception-only / 43% mixed / 1% reasoning-only).
- Xiao et al., *NExT-GQA*, CVPR 2024; Yi et al., *CLEVRER*, ICLR 2020; Masry et al., *ChartQA*, Findings of ACL 2022.
- Bai et al., *Qwen3-VL Technical Report*, arXiv:2511.21631; Zhu et al., *InternVL3*, arXiv:2504.10479; Zhang et al., *Penguin-VL: Exploring the Efficiency Limits of VLM with LLM-based Vision Encoders*, arXiv:2603.06569 (Tencent AI Lab).
