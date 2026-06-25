# N1 PREREGISTRATION — constructive (B2): can a curriculum force a *load-bearing* internalized chain?

> **Committed BEFORE running** (2026-06-25). Predictions, dataset, curriculum filters,
> the probe, and the decision rule are fixed here; results go in a later
> `n1_complete_*.md`. Honors chapter_plan red-line "先写预注册 verdict 再看结果" and the
> two-directions requirement. This draft is **shaped by N3's findings** (see §1).

## The question N1 answers
The diagnostic half of the paper shows distillation transfers **reading**, not a
load-bearing chain: across charts, tables, and natural images the model re-reads /
reconstructs the answer and ignores its CoT's specific numbers (F≈0, follow≈0).

N1 is the **constructive (B2)** counter-test:

> If we build a training curriculum **only from problems where the answer cannot be
> obtained by re-reading a single value — it genuinely requires a multi-step arithmetic
> chain — does the SFT'd student then carry a *load-bearing* chain** (corrupting an
> on-path intermediate flips the answer, `follow`↑, `other` flat), or does the
> read/reconstruct shortcut survive even this?**

Either way is a result: success = "distillation *can* internalize load-bearing reasoning
when the curriculum removes the shortcut" (a constructive positive); the more-likely
outcome (per §1) = "even a shortcut-removing curriculum doesn't make the chain
load-bearing" → a hard, well-supported **boundary** on what trajectory distillation can
internalize, with masked + N3 + N1 as three converging pieces.

## 1. How N3 shaped this design (the four changes ②③④⑤)
N3 (natural-image probe) returned `follow`≈0 / `other` flat **even when the answer-bearing
number was falsified and the image masked** — because the answer was **redundantly
recoverable** from the chain's surviving parts. Lessons baked into N1:

- **② Redundancy-aware, *targeted* probe (not random corruption).** "follow↑" is
  ambiguous unless the corrupted number is provably *on the causal path* and *not
  recoverable* from the rest of the chain. N1 uses FinQA's **gold program** as an oracle:
  corrupt a numeric operand that, **re-executed, actually changes the final answer**, and
  whose value is **not redundantly restated**. We corrupt *that* token, not a random one.
- **③ Curriculum closes BOTH escape routes** (re-perception AND chain-redundancy): keep
  only problems that are (a) ≥2-op multi-step, (b) combine ≥2 distinct table values,
  (c) have ≥1 corruption-flippable operand, (d) whose answer is **not equal to any single
  table cell** (not single-cell-readable), (e) all operands present in the table (so the
  rendered image is self-sufficient → bypass is *possible*, making a load-bearing result
  meaningful).
- **④ Expected failure signature shifts.** N3 showed deprivation → reconstruct, not
  hallucinate. So the predicted N1 failure is **`snap` stays high / `follow`≈0 (bypass
  persists)**, NOT `other`↑. Decision rule updated accordingly (§4).
- **⑤ Teacher = text reasoner reading the GT table (primary).** To inject a *provably*
  load-bearing chain, the teacher reads the table as **text** (no image to re-read) and
  must compute. A deterministic **gold-program→NL** teacher is the robustness arm (removes
  teacher-quality confound; provably load-bearing by construction).

## 2. Dataset — FinQA (chosen after a web search; rationale)
Surveyed TabMWP / FinQA / TAT-QA / MultiHiertt / chart-VQA. **FinQA** wins because it is
the only one giving, per example, a **gold executable reasoning program** — which is
exactly the ② oracle: parse → execute → verify the answer, and verify *corruption-
flippability* of each operand by re-execution. Plus: free-form **numeric** answers (same
relaxed grader, no MCQ), table available as **text** (for the ⑤ teacher), genuinely
multi-step, within the paper's "table QA" scope, and deeper arithmetic than TabMWP's
mostly 1–2-step problems.

- Source: original FinQA json **with `program`** (`dataset/{dev,test}.json`), fetched
  from the jsDelivr GitHub CDN (no proxy). `data/distill/finqa/raw/{dev,test}_full.json`.
  (HF auto-parquet drops `program`; train.json exceeds the CDN 20MB cap — not needed.)
- **Splits**: **dev (883) → curriculum-train**, **test (1147) → eval-probe** (disjoint,
  both program-backed). Tables rendered to images `finqa_<i>.png` for the VLM student.
- **Validated executor + yield** (`scripts/build_n1_curriculum.py`, ops add/subtract/
  multiply/divide/exp, `#k` refs, `const_`): executor reproduces `exe_ans` ~96%; under the
  ③ filters → **dev 196 / test 262** fully-qualified load-bearing-chain problems.

## 3. Operationalization (fixed)
- **Curriculum-train set** = FinQA dev filtered by ③ (n≈196 → after teacher consistency
  ≈150–180). Each record: `{image_path(rendered table), question, cot(teacher),
  answer, gold, program, load_bearing_values, n_ops}`.
- **Control = vanilla-SFT**: SFT on a **matched-size random FinQA dev sample** (NO ③
  filter — includes single-step / single-cell-readable problems), same teacher, same n,
  same image rendering. Isolates the *curriculum* as the only difference.
- **Teacher (⑤)**: primary = text reasoner (DeepSeek) reads table-as-text + question →
  CoT + answer; consistency-filter (final == gold) AND require the CoT to contain ≥1
  load-bearing operand value (so the probe has a target). Robustness arm = deterministic
  gold-program→NL chain (no API, provably load-bearing).
- **Student**: Qwen3-VL-8B (primary; 32B if feasible), LoRA SFT via `poc_sft.py` on the
  rendered-table images. Same recipe for B2 and vanilla.
- **Probe (②), on the held-out test set** (`battery_n1_targeted.py`, to be written):
  for each test problem the student answers correctly with a CoT, identify a number in the
  student's CoT that matches a **gold-program operand verified corruption-flippable**;
  corrupt **that** token (vs shuffle control), force-continue, classify
  **snap / follow / other** (`battery_followrate.py` 口径). Report B2 vs vanilla vs base.

## 4. Preregistered predictions (BOTH directions reported)
Baseline (from N3, base/SFT students, present): `follow`≈0, `snap` high.

- **H_success** (constructive positive): B2-student image-present **`follow`↑ vs vanilla-SFT
  and vs base, with `other` flat** — the curriculum made the chain load-bearing.
- **H_fail_bypass** (judged **most likely**, per N3): `follow` stays ≈0 and **`snap`
  stays high** — the read/reconstruct shortcut survives the curriculum; B2 ≈ vanilla.
- **H_fail_fabricate** (less likely, per N3's flat-`other`): `other`↑ instead of `follow`↑.

Priors: N3 showed *test-time* deprivation (masking, natural images) does not force a
load-bearing chain; a *training-time* curriculum is a stronger lever, but bypass is the
default, so H_fail_bypass is the base rate and H_success would be the strong, surprising
positive.

## 5. Decision rule / acceptance
- [ ] Curriculum-train (dev, ③-filtered) + matched vanilla set built, rendered, teacher
      CoT + consistency-filtered; load-bearing operands tagged per example.
- [ ] B2-student and vanilla-student SFT'd (8B; 32B if feasible).
- [ ] Targeted ② probe on test: **snap/follow/other for B2 vs vanilla vs base**, with the
      shuffle control and per-finding flippability verified via the gold program.
- [ ] Verdict written for whichever of H_success / H_fail_bypass / H_fail_fabricate holds;
      report Δ(follow, other, snap) B2−vanilla with binomial CIs. **Either way a result.**

## 6. Scope / limitations (method rigor, recorded up front)
- Targeted corruption uses the **gold program** to guarantee the corrupted operand is
  on-path and non-redundant — this is the rigor N3 showed is required; it is a *lower
  bound* on load-bearing-ness only in the benign sense that we test one verified operand.
- FinQA tables rendered to images: perception difficulty is higher than ChartQA/TabMWP
  (good — re-deriving at answer time is costly, the intended pressure), but too-hard cases
  fall out of the student's correct-with-CoT kept set; report kept-n and base acc.
- Curriculum from dev (n≈196) is modest; if the kept SFT set is small, augment by pulling
  full FinQA train (mirror, when bandwidth allows) — does not change the design.
- Student is a VLM that *can* re-read the table image (operands are all in-table by filter
  (e)); this is deliberate — it makes a load-bearing result non-trivial and a bypass
  result the honest default.
- Red-line preserved: we do not claim corruption alone proves a chain; shuffle control +
  the gold-program flippability verification + B2-vs-vanilla contrast carry the verdict.
