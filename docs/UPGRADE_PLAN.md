# Upgrade Plan — *Reading, Not Reasoning* → AAAI-submittable

> **Goal (reset 2026-06-23):** ship an **AAAI** submission — **abstract 7/21, full paper 7/28**.
> Companion doc: [paper/chapter_plan.md](paper/chapter_plan.md) (the章节骨架 + INSIGHT + 红线).
> This file is the **detailed experiment spec**; [../todo/0622.md](../todo/0622.md) is its checkbox list.
>
> **Posture (changed from the 0622 "defend existing draft" plan):**
> **硬化审计为脊 + 嵌入小建设性 (B2) + 拔高版 regime-依赖组织性发现。**
> - **Spine** = the hardened causal faithfulness audit, entirely on **static charts + tables**
>   (ChartQA + TabMWP) — bypasses the video-ingest segfault.
> - **Constructive (N1 / B2)** = a *curriculum* training intervention that **pre-registers** whether
>   a load-bearing chain can be installed; result is informative **either way**.
> - **拔高版 (N3 / N4)** = the load-bearing locus is **regime-dependent** — chart/table = perception
>   re-read load-bearing (CoT decorative); natural-image (small V* probe) = textual CoT load-bearing
>   (the *mirror* of 2510.23482), predicted by the perception bottleneck.
> - **Video** results → **appendix** (boundary motivation only).
>
> **Conventions (unchanged):** every number flows through the fingerprinted result store
> (`data/distill/results/results.jsonl`); CLEAN is primary; paired-bootstrap CI is the headline test,
> McNemar the SFT sanity test. Each work unit is self-contained:
> **Purpose / Claim it defends / Build / Run / Acceptance / Expected / Risk.** Do them in P0→P4 order.
>
> **Two software-confirmed enablers this plan rests on (unchanged):**
> 1. `run_chartqa_gate.py` and `poc_causal_probe_32b.py` both consume `{case_id, question, gold}` rows
>    + an `--img-dir`. **Scaling n = a bigger rows-file + image dir; core code untouched.**
> 2. ChartQA **test** split is disjoint from the **train** images used for the 150-CoT teacher set →
>    SFT train/test stays clean when we pull more test items.

## Soft points this plan closes (now five, not three)
- **Power.** ChartQA n=60; 32B PoC McNemar **p=.125** (4 discordant pairs). → WU-1.
- **Single-dataset.** Positive+causal half rests on **ChartQA alone**. → WU-3 (TabMWP, now core).
- **Two-intervention probe.** → WU-2 battery **+ N2 re-perception control** (the new named control).
- **No headline figure.** → WU-5 (faithfulness⊥accuracy scatter + regime heatmap).
- **NEW — novelty/collision (existential).** Collision check **done** (16 papers downloaded to
  `paper_references/`, 7 read in full). Verdict: the **distilled single-forward, no-tool, discrete-CoT
  chart/table** cell is **open**, but five papers must be defended against in §2/§4:
  - **Threats** (cite + differentiate, see chapter_plan redlines): **2510.23482** (tool-using MCoT,
    *visual* ignored — opposite pole), **2602.22766** (ICML'26; *latent* tokens, degeneracy),
    **CodeV 2511.19661** (audits *tool-output crops*, by its own statement **not** CoT tokens).
  - **Method critiques** (pre-empt in §4 "Robustness to known confounds"): **2502.14829**
    (context-perturbation only measures *contextual* faithfulness → our re-perception + image-mask
    **identify** the reconstruction source = the chart, converting the confound into the finding);
    **2402.14897** (Lanham metric is *disguised accuracy* under option bias → our substrate is
    **open-ended numeric**, no letter options; report accuracy beside every flip-rate).
  - **Redlines:** do NOT claim the *corrupt* intervention itself as novel (Lanham/2510.23482 own it);
    do NOT use "internalized reasoning is not load-bearing" as the abstract's main clause (2602.22766
    owns it). Lead with **reading-not-reasoning + re-read mechanism + regime-dependence**.

-----

## P0 — Prerequisites (do today; gates everything)

- Start serving in `vllm-qwen`: **32B @:30001, 8B @:30002** (`serve_vlm_32b.sh`; 8B TP=2). Smoke each
  endpoint. (Serving is currently DOWN per §0.2.)
- Build the two test sets (WU-1.1 + WU-3): `build_chartqa_test.py`, `build_tabmwp_test.py`.
- All non-serving commands use `conda run -n mbe-up …` (env ready, §0 done).

-----

## WU-1 — Scale ChartQA evaluation n (60 → 300+) [THE power fix]

**Purpose.** Resolve the underpowered 32B PoC (McNemar p=.125) and the "effect < n can resolve" kill.
Static images → bypasses the video-ingest segfault. Highest-leverage upgrade.

**Claim it defends.** §5 (the regime-2 SFT gain) and §4/§6 causal-probe n's. Converts "positive but
underpowered" into "properly powered" (positive OR clean null — both publishable, now defensible).

**Build.**
- `scripts/build_chartqa_test.py` (NEW): pull `N=400` from HF ChartQA **test** (human + augmented,
  record ratio); download via clash proxy one at a time; images →
  `/home/gpus/mbe_data/chartqa_test_images/chartqa_<i>.png`; rows →
  `data/distill/chartqa/test_cases_400.jsonl` as `{case_id:"chartqa-<i>", question, gold}`.
- Assert hash-disjoint from the 187 train images; log count. Run the free §0.5 relaxed-numeric regrade
  (record params + letter-luck, expected 0 for open-ended).

**Run.**
- Gate at scale: `run_chartqa_gate.py --model-id {32b,8b,4b} --dump …test_cases_400.jsonl --img-dir … --methods self_reflect orch_reflect_blind --seeds 5 --concurrency 8`.
- SFT eval at scale: point per-epoch held-out eval in `poc_sft_32b_qlora.py` (+ `poc_sft.py` for 8B)
  at `test_cases_400.jsonl`. **Keep the 150-CoT train set fixed** (isolate eval-n). Re-select inflection.
- Causal probe at scale (folded into WU-2 run).

**Acceptance.**
- [ ] `test_cases_400.jsonl` (n≥300 after grading), image dir, hash-disjoint, in repo.
- [ ] 8B + 32B SFT Δacc at n≥300 with bootstrap CI **and** McNemar p (32B now on ~15–25 discordant pairs).
- [ ] power table updated (`power_table.py`): min detectable net at new n.

**Expected.** Either 32B reaches significance (strengthens §5) or holds as a clean high-power null
(makes the §6 "reading not load-bearing reasoning" verdict *stronger*). Probe direction
(shuffle ≥ corrupt, present) tightens with n.

**Risk.** HF throttling — pull in batches; n=300 suffices (min detectable ≈10%).

-----

## WU-2 — Faithfulness battery + N2 re-perception control [audit depth]

**Purpose.** Move from 2 interventions to a **battery** so the "not load-bearing" verdict is multiply
confirmed, and add the **re-perception control** — the strongest differentiator vs all prior probes,
and the answer to the 2502.14829 critique.

**Claim it defends.** §4 (the instrument) + §5 (the central verdict). Each lever is an independent test.

**Build.** Extend `poc_causal_probe_32b.py` (+ `poc_causal_probe.py`) alongside existing
`corrupt_number` / `shuffle_cot`:
- **N2 `re_perception` (NEW, the headline control):** corrupt a numeric intermediate, then check
  whether the answer **snaps to the TRUE chart value** (→ re-read, reading-not-reasoning) or follows
  the **injected wrong value** (→ load-bearing). Report the snap-rate. Directly defuses 2502.14829.
- `truncate_cot(cot, frac∈{.25,.5,.75})` — early-answering (most diagnostic).
- `delete_steps(cot, k)` — progressive deletion; flip-rate vs k curve.
- `paraphrase_cot(cot)` — DeepSeek API, **cache to file** for offline re-runs (corrupt's two-sided control).
- `filler_cot(cot)` — Pfau length-matched filler (shuffle's control; rules out length/format confound).
- `--interventions reperception truncate delete paraphrase filler` (default all); reuse `--mask-image`.

**Run.** Full {8B,32B} × {present,masked} × {intervention} grid at WU-1's scaled n.

**Acceptance.**
- [ ] battery table: intervention × {8B,32B} × {present,masked} flip-rate, n≥300.
- [ ] re-perception snap-rate reported (the reading-not-reasoning direct evidence).
- [ ] early-answering curve; paraphrase = corrupt control, filler = shuffle control.
- [ ] **accuracy reported beside every flip-rate** (defuses 2402.14897 disguised-accuracy).

**Expected.** Present: corrupt low-flip, paraphrase low-flip, early-answer flat, **re-perception snaps
to true value** → re-read confirmed by ≥4 levers. Masked: corrupt/delete flip rises → latent chain exists.

**Risk.** Paraphrase API mid-probe — cache for determinism.

-----

## N1 — Constructive (B2): can a load-bearing chain be installed? [pre-registered]

**Purpose.** The one constructive contribution. Diagnosis: distilled CoT transfers reading because the
model re-reads. **B2 = curriculum** that removes the re-read shortcut at *training* time, forcing chain
reliance. **Pre-registered**: result informative either way (chapter_plan `prereg` INSIGHT).
**(Rejected B1 = masked-image-augmented SFT — it risks training hallucination; the user's call.)**

**Claim it defends.** §6 (constructive). A positive = a recipe for faithful distilled CoT; a null =
"the perceptual shortcut is *deeply preferred*" = strengthens the audit thesis.

**Build / Run.**
- Curriculum filter on the teacher CoT set: keep ONLY problems where **re-reading a single cell cannot
  yield the answer** (genuine multi-step arithmetic over ≥2 reads). Target ~150 consistency-filtered.
- SFT the small VLM on this set (`poc_gen_cot.py` filter → `poc_sft*.py`); merge/serve or in-proc.
- Run the WU-2 battery on the student. **Primary judge:** image-present `corrupt-flip` (and re-perception
  snap-rate) of B2-student **vs vanilla-SFT student**.
- **Alt arm (optional, from demoted WU-4.1):** a **text-reasoner-over-GT-table** teacher (DeepSeek over
  the GT table → provably load-bearing chain) → distill → battery. Tests teacher-side vs student-side.

**Acceptance.**
- [ ] B2-student battery vs vanilla-SFT, n≥300; corrupt-flip + re-perception snap-rate Δ reported.
- [ ] pre-registered verdict written **before** looking (`prereg` INSIGHT), then the result.

**Expected (pre-registered).** If B2 corrupt-flip not significantly > vanilla-SFT → **perceptual
shortcut is deeply preferred** (audit thesis ++). If it *is* > → a constructive recipe for faithful CoT.

**Risk.** Net-new + training time → **start early (P2)**. Pre-registration removes the "must be positive" pressure.

-----

## WU-3 — TabMWP: the second reasoning-bound dataset [kills n=1-dataset; now CORE]

**Purpose.** The causal verdict must be cross-task. TabMWP = multi-step arithmetic + re-readable table
+ ships GT table (also feeds N1's alt arm). Static → no ingest blocker.

**Build/Run.** `build_tabmwp_test.py` mirrors `build_chartqa_test.py` (`{case_id,question,gold}` + img
dir); reuse `run_chartqa_gate.py` (dataset-agnostic) + `poc_*`/probe with new paths. Full ladder
(4B/8B/32B free+orch, multi-seed) for the MAP; then SFT + WU-2 battery on the regime-2 base.

**Acceptance.**
- [ ] TabMWP on the MAP (regime per cell, net±CI).
- [ ] ≥1 regime-2 cell with SFT Δacc + battery → does "reading not reasoning" replicate off ChartQA?

**Expected.** Verdict replicates → thesis generalizes. If TabMWP shows a *load-bearing* chain where
ChartQA did not → an even better finding (boundary depends on full re-readability) — report it.

**Risk.** Scope creep. Cap at ONE dataset; if tight, just the regime-2 cell + probe.

-----

## N3 — Natural-image V* probe: the regime-dependence pole [拔高版; N4 organizing finding]

**Purpose.** Demonstrate **both poles in one paper**: on natural-image fine-grained perception, the
*textual* CoT is load-bearing and perception is the bottleneck — the **mirror** of the chart finding
(and of 2510.23482). The perception bottleneck predicts which pole.

**Claim it defends.** §5.2–5.3 (the organizing finding — the拔高 point that lifts the paper above "yet
another faithfulness audit").

**Build/Run.** A **small** V* subset → `{case_id,question,gold}` + img dir (reuse the probe harness).
Run free-form + the WU-2 battery (corrupt/shuffle/re-perception, present+masked) on a distilled/base VLM.
Keep scope small —佐证双极, not a full study.

**Acceptance.**
- [ ] V* probe: corrupt/shuffle flip + re-perception, showing **textual CoT load-bearing** (opposite of charts).
- [ ] regime axis stated: perception-bottleneck (re-readable-from-image?) predicts the load-bearing pole.

**Expected.** V* (small targets in high-res, *not* re-readable) → CoT load-bearing; charts (re-readable)
→ CoT bypassed. The contrast = the organizing finding.

**Risk.** Scope creep — keep V* small; it佐证, doesn't carry the paper. (Out-of-scope: full natural-image study.)

-----

## WU-5 — Faithfulness ⊥ Accuracy axis + figures [the quotable contribution]

**Purpose.** Reframe around figures: accuracy gains and faithfulness are decoupled, and the load-bearing
locus is regime-dependent.

**Build.** `scripts/build_faithfulness_axis.py` (NEW): `F = flip_corrupt − flip_shuffle` (present)
and/or masked−present gap; join SFT Δacc → `results/faithfulness.json` + figures:
- **Faithfulness–accuracy scatter** (x = SFT acc gain, y = F; one point per cell/teacher) — the punchline.
- **Regime heatmap** (dataset × model, colored by regime, net±CI + oracle; + the V* pole) — the §8/§5.3 fig.

**Acceptance.**
- [ ] both figures from the result store (no hand-copied numbers), in the paper.
- [ ] faithfulness column added to the master table.

**Expected.** Regime-2 cells sit high-accuracy / low-faithfulness; the two poles separate on the regime fig.

-----

## WU-4 — Cross-teacher control [DEMOTED → optional / appendix]

**Status.** Demoted from the 0622 "highest-value extension." **4.1 (text-reasoner-over-GT-table teacher)
is folded into N1 as the optional alt arm.** 4.2 (non-Qwen VLM teacher, InternVL3-8B @:30003) → appendix
if time allows; confirms the result isn't Qwen-CoT-specific. Do only after P0–P3 land.

**Acceptance (if run).** [ ] student from a non-Qwen teacher; battery; teacher×faithfulness row in store.

-----

## WU-6 — Positioning & writing [parallel; now carries the 5-paper defense + reframe]

**Purpose.** Reframe the whole paper (N4) and install the novelty defense — no new experiments.

**Build.**
- §2 Related Work (draft already assembled, see chat / chapter_plan): four paragraphs —
  internalization (acc-only: VPD/Zooming-without-Zooming/PEARL/ChartPaLI/Chart-R1; LOCUS concurrent);
  faithfulness lineage (Lanham/Turpin/Pfau) + multimodal (2510.23482/CodeV/2602.22766) + critiques
  (2502.14829/2402.14897); our position (four-axis distinction); chart motivation
  (ChartMuseum/PerceptionBottleneck/CharXiv/CHART-NOISe; TabMWP).
- §4 "Robustness to known confounds" paragraph (drafted) — defuses 2502.14829 + 2402.14897.
- Abstract around the WU-5 decoupling figure + regime-dependence; **honor the two redlines**.
- `research-paper-writing` skill for the Related Work + abstract pass.

**Acceptance.** [ ] §2 cites the 5 defense papers + lineage; §4 confound paragraph in; abstract → headline
figure + regime; power sentence at new n; redlines honored.

-----

## Execution order (P0→P4), deadline-aware

- **P0 (today):** serving up (32B/8B) + build both test sets.
- **P1 (2–3 d, critical path for the abstract number):** WU-1 (scale n) → WU-2 + N2 battery/probe at n≥300.
  *After P1 the headline number is locked.*
- **P2 (start early, net-new + training):** N1 (B2 curriculum) → SFT → battery.
- **P3:** WU-3 (TabMWP) ∥ N3 (V* pole).
- **P4:** WU-5 figures; WU-6 writing (parallel throughout); WU-4 only if time remains.
- **Abstract 7/21** needs P1 (+ ideally N3 for the regime claim). **Full 7/28** needs P1–P4.

**Submission floor (if tight):** P1 + N3 + WU-5 + WU-6 = a defended audit + the regime figure.
N1 (B2) is the one risk; pre-registration兜底.

**Paper-section mapping (→ chapter_plan):** WU-1→§3/§5; WU-2+N2→§4; N1→§6; WU-3→§5; N3→§5.2–5.3;
WU-5→figures+§5; WU-6→§1/§2; WU-4→appendix; video→appendix.

**Out of scope (revisit after acceptance):** video-n scaling (BGE/SigLIP ingest segfault); full
natural-image study (N3 stays small); synthetic perception×reasoning generator; mechanistic/attention
layer (attention-knockout / activation patching / logit-lens — the cheap one可作 camera-ready stretch).
