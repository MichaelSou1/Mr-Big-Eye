# Supplementary Experiment Spec — Internalization Boundary Result

> Goal of this document: turn the exploratory `progress.md` log into a defensible,
> publishable boundary result. This spec is written to be executed incrementally with
> Claude Code (CC). Each section is a self-contained work unit with: purpose, what to
> build/run, exact acceptance criteria, and the claim it defends. Do them in order;
> §1 gates everything else.
> 
> Central contribution this spec must defend (reframed 2026-06-09 — from a flat
> "no headroom" to a **decomposition + map**, which is a stronger, more positive
> framing for an application-grade artifact):
> 
> **A perception-vs-reasoning decomposition for VLM "agentic reasoning" evaluation that
> MAPS where internalizing tool-free reasoning is feasible. The map has two regimes:
> (1) perception-/selection-bound (short-video, evidence-not-reliably-in-sample) — here
> agentic/reflective reasoning has NO statistically reliable internalizable headroom over
> a single forward pass, across scales and families; the binding constraint is perception
> fidelity, not reasoning; the lone observed positive (8B +8%) is shown to be within
> run-to-run variance. (2) reasoning-bound (perception solved by a strong-enough base,
> e.g. 32B on static charts/numerical) — here residual failures ARE multi-step reasoning,
> and internalization has headroom (optionally demonstrated by a small SFT PoC, §11).**
> 
> The negative boundary (regime 1) and the positive lead (regime 2) are two halves of the
> same map; the reusable, citable artifact is the decomposition protocol (frames-visible
> partition + perception-headroom ladder + oracle-perception upper bound + blind/sighted
> critic ablation), not any single accuracy number. The +8% retraction is a methodological
> contribution about VLM-reasoning eval variance.
> 
> Non-negotiables carried from the original project:
> 
> - Inference target is tool-free single forward (type-1 only).
> - Trajectory/reflection source may be a larger text model; the VLM backbone under test
>   is the small one.
> - Vision-only unless a section explicitly says otherwise.
> 
> **Do not pre-commit to "net ≈ 0 everywhere."** Exp-8 already shows the boundary does NOT
> hold uniformly (32B-ChartQA is reasoning-bound). The result is a MAP; let the data place
> each (dataset, base) cell into regime 1 or 2. A clean map beats a forced uniform negative.

-----

## §0. Definitions and global conventions (CC: read first, do not skip)

These fix the vocabulary so every later result is comparable. Put these in
`app/distill/eval_common.py` as constants/helpers and import everywhere.

- **Setting**: a (dataset, base_VLM, method) triple. `method ∈ {free_form, self_reflect, orch_reflect_blind, orch_reflect_sighted, agent_retrieval}`.
- **free_form**: single tool-free forward pass over the fixed uniform-N frames. The
  baseline everything is measured against.
- **self_reflect**: multi-turn (2-turn) tool-free self-re-examination by the same VLM over
  the SAME frames.
- **orch_reflect_blind**: 30B text model orchestrates re-checking; it does NOT see frames,
  only the VLM’s text reads. (This is what the log called “orch”.)
- **orch_reflect_sighted**: NEW. Same as above but the 30B/critic is itself a VLM that sees
  the frames. Disambiguates “critic is blind” from “VLM perception is the wall”. (§4)
- **agent_retrieval**: full agent with frame/region retrieval (type-2 lever included). Used
  only to show retrieval is net-negative in this regime; not a main claim.
- **net**: `acc(method) − acc(free_form)` on the SAME case set, SAME frames.
- **gain / lost case**: a case where method is right & free_form wrong (gain), or method
  wrong & free_form right (lost). Report both, not just net.
- **N_frames**: fixed uniform sample count. Default 16. Must be identical across methods
  within a setting (this is the controlled variable — frame selection is held constant).
- **Seed**: controls (a) frame-sampling jitter if any, (b) decoding sampling, (c) case
  shuffling. Every stochastic knob keyed off one integer seed.

**Determinism rule:** record, per run, the full config fingerprint: dataset split hash,
N_frames, model id + revision, decoding params (temp, top_p, max_tokens), seed, prompt
hash, code version. Reuse the existing fingerprint mechanism from `eval_fingerprint.py`.
A result without a fingerprint is not a result.

-----

## §0.5. Label-noise & grader audit — THE OTHER GATE (do BEFORE §1; it sets the floor under the floor)

**Why this is critical and was missing.** Per Exp-8 per-case reading, on NExT-GQA roughly
**40% of free-wrong cases are label ambiguity** (e.g. gold "feeding" vs model "grazing";
"excited" vs "enjoying the music"; near-duplicate MCQ distractors), i.e. ~10–15% of ALL
cases have a debatable/ambiguous gold, plus ~4% are **grader artifacts** (MCQ scored by
option LETTER while the answer TEXT disagrees — "letter-luck"). The headline effects are at
the **±2%** scale. **A label/grader noise of 10–15% sits an order of magnitude above the
effect you are trying to measure.** Two consequences:

1. A reviewer kills the paper with one line: *"your effect is smaller than your label noise."*
2. Ambiguous cases churn between two near-synonymous options across seeds/methods, which is a
   large hidden contributor to the "run-to-run variance" §1 attributes to decoding. **Cleaning
   labels lowers `σ_freeform`, raises power, and makes "powered to see 2%, saw nothing"
   *stronger*.** This is not defensive bookkeeping; it tightens the core result.

### §0.5.1 Fix the grader first (cheap, deterministic)
- Replace pure-letter MCQ matching with **text-aware matching**: a case is correct only if the
  selected option's TEXT (not just its letter) matches gold. Flag every case where letter and
  text disagree; treat letter-only matches as **incorrect** (or exclude + report count).
- Reuse `app/mcq.py` (`parse_candidates`, `selected_candidate`) and the dump artifacts in
  `data/distill/analysis/dump_*.jsonl` (which already store the full answer TEXT).

### §0.5.2 Audit label ambiguity, build a clean-label subset
- On each MAIN dataset, adjudicate a sample (≥ 150 cases, or all if small): for every case mark
  `{clean | ambiguous | wrong-gold}`. Do it with a strong LLM-judge given (question, options,
  gold, a short rationale) AND spot-check ~30 by hand to estimate judge error.
- Report the **label-noise rate** with a CI. Then define `CLEAN = cases judged clean`.
- **All headline accuracies and nets (§1, §3, §4, §5) are computed on CLEAN.** Report the full
  set too, but CLEAN is the primary. State n_clean per dataset.

### §0.5.3 Acceptance
- [ ] text-aware grader replaces letter-matching everywhere; letter-luck count reported.
- [ ] label-noise rate (with CI) measured on each main dataset; clean-label subset defined.
- [ ] every headline number reported on CLEAN (primary) and full (secondary), with n_clean.

**Expected payoff:** σ_freeform on CLEAN is meaningfully below the full-set σ; minimum
detectable effect (§2) improves; the boundary claim is stated on labels you can defend.

-----

## §1. Variance characterization — THE GATE (do this AFTER §0.5; nothing else counts until done)

**Why first:** the +8% retraction proved single-run nets are not trustworthy at the ±2%
scale you’re working in. Until you know the noise floor, you cannot tell signal from
variance for ANY number in the log. This section establishes the noise floor and re-tests
every existing net against it.

**Methodology fix (avoid inflating your own noise floor).** There are TWO distinct variance
sources; do not blend them:
- **Case sampling variance** — finite n. Quantify with **paired bootstrap over CASES** (§1.3).
  This is the PRIMARY CI for every headline number and applies even to deterministic methods.
- **Decoding/orchestrator stochasticity** — only for methods that sample (temp>0). `free_form`
  should be run at **greedy (temp=0) → deterministic, no seed variance**; its only uncertainty
  is case-sampling (bootstrap). `self_reflect`, `orch_reflect_*` involve a temp>0 critic /
  multi-turn generation, so they DO need the K-seed loop. Running a 10-seed loop on greedy
  free_form just measures zero and wastes compute; the +8% retraction comes from the
  *orchestrated* method's stochasticity, which is exactly where the seed loop belongs.

### §1.1 Noise floor of `acc(free_form)` (greedy + bootstrap; on CLEAN)

- Main dataset = NExT-GQA causal/temporal **CLEAN** subset (§0.5); main base = 8B, repeat 4B.
- Run `free_form` once at **temp=0**, n ≥ 200 (CLEAN). Compute the 95% CI of accuracy by
  **bootstrap over cases** (B=10,000). This CI width is `σ_case` — the irreducible
  finite-sample floor at this n.
- For the stochastic methods, ALSO run K=10 seeds (temp>0) and report per-seed std `σ_decode`.
- **Noise floor used for verdicts = the bootstrap CI of the NET** (§1.3), which already folds
  in both sources for paired comparisons. Report `σ_case` (free_form) and `σ_decode` (orch)
  separately for transparency.

### §1.2 Re-test every existing net against the noise floor

For each setting that produced a headline number in `progress.md`
(`self_reflect`, `orch_reflect_blind`, `agent_retrieval`, per dataset, per model), re-run
across **K = 10 seeds** with paired evaluation (same cases, same frames as its free_form
counterpart per seed).

- Report, per setting: `net_mean`, `net_std`, 95% CI, and a paired significance test (§1.3).
- **Explicitly re-test the 8B +8% case** and document that it crosses zero. This becomes a
  named result in the paper (“the lone positive does not survive multi-seed evaluation”).

### §1.3 Statistical test (fix one, use everywhere)

Because outcomes are per-case correct/incorrect and methods are evaluated on the SAME cases
(paired), use **paired bootstrap over cases** as the primary test:

- For each of B = 10,000 bootstrap resamples of the case set, compute net; report the
  fraction of resamples with net ≤ 0 (and ≥ 0). 95% CI = 2.5/97.5 percentiles.
- Secondary / sanity: **McNemar’s test** on the (gain, lost) discordant pairs, per seed,
  then aggregate.
- Decision rule for the paper: a setting (on **CLEAN**) shows a real effect only if the 95%
  paired-bootstrap CI of net **excludes 0**. Report it as `effect` / `within-variance`
  accordingly. For stochastic methods, average net over the K=10 seeds before the per-case
  bootstrap, OR report the seed-mean net with a CI that includes seed variance (bootstrap
  cases × resample seeds). Always report gain and lost counts, not just net.

### §1.4 Acceptance criteria for §1

- [ ] Free_form noise floor (`σ_case` bootstrap CI) on CLEAN for 4B & 8B, n ≥ 200.
- [ ] Every existing headline net re-run on CLEAN with paired bootstrap CI (stochastic methods
  also K=10 seeds for `σ_decode`).
- [ ] Results table: each setting → free_form acc, net_mean ± 95% CI, verdict
  {effect | within-variance}, on CLEAN (full set as secondary column).
- [ ] The 8B +8% explicitly shown as within-variance (its CI crosses 0), with the table row.

**Likely outcome (state it honestly whatever it is):** most nets collapse into
“within variance,” and the true claim becomes “net ≈ 0 everywhere (no reliable headroom),”
which is a CLEANER boundary result than “uniformly negative.” Do not fight the data.

-----

## §2. Power & sample-size sanity (cheap, prevents a reviewer one-shot kill)

**Purpose:** show you can detect an effect if one existed — otherwise “no effect” is
unconvincing (absence of evidence vs evidence of absence).

- Given `σ_freeform` from §1, compute the **minimum detectable net** at n = 200, 300, 500
  with 80% power, α = 0.05 (standard paired-proportion power calc; a short script in
  `scripts/power_analysis.py`).
- Report a sentence like: “At n = 300 we can detect a true net ≥ X% with 80% power; observed
  |net| < X% across all settings.” This converts “we saw nothing” into “we were powered to
  see Y and saw nothing.”
- If the minimum detectable effect is large (e.g. > 5%) at your current n, **increase n**
  until you can detect ~3%. NExT-GQA has 1570 clips; you only ingested 70–120. Scale ingest
  (§7) to lift n where it matters.

**Acceptance:** [ ] power table (n × detectable-effect) in repo; main results run at an n
that gives ≤ ~3% minimum detectable effect.

-----

## §3. The mechanism claim — perception fidelity is the wall (this is your strongest asset)

**Purpose:** the boundary result is only interesting if you explain WHY. The mechanism is
“perception-bound, not reasoning-bound.” §1 shows no headroom; §3 shows the cause. Make the
perception-headroom probe a **uniform, cross-dataset protocol**, not a one-off (currently it
only exists for NExT-GQA / Exp-C).

### §3.0 Frames-visible partition (do this FIRST — it makes the central claim honest)

The claim is about the regime "answer-bearing frames are *already visible*." But uniform-N
does NOT always contain the decisive frame — when it misses, the case is **frame-selection
(type-2) limited**, not a fair test of reasoning headroom. So partition every case BEFORE the
headroom analysis, using grounding GT:

- `EVIDENCE_IN`: ≥1 uniform-sampled frame falls inside the GT grounding window (within
  tolerance). The genuine **frames-visible** regime. (Reuse `app/distill/frames.py:covers_evidence`.)
- `EVIDENCE_OUT`: no sampled frame in the GT window → selection-limited, reported separately.

**The reasoning-headroom result (§1 nets) is stated primarily on `EVIDENCE_IN ∩ CLEAN`** —
the cells where frames really are visible and labels are defensible. `EVIDENCE_OUT` is reported
as the type-2/selection slice (the agent's only edge, non-internalizable). This partition is
itself a methodological contribution; datasets without grounding GT (ChartQA static = whole
image is the "frame") are trivially all `EVIDENCE_IN`.

### §3.1 Standardize the perception-headroom probe

Define `diag_perception_headroom` as a reusable protocol (refactor the existing NExT script):
For the set of free-wrong cases, re-run free_form under escalating perception aid:

1. uniform-N @ native res (control)
1. uniform-N @ high res (e.g. 640 / 768 longest side)
1. **GT-localized frames** @ high res (the gold evidence window, max res) — requires the
   dataset to have temporal/region GT; NExT-GQA has grounding GT.
1. (image datasets) GT-region crop @ high res.
1. **ORACLE perception** (the upper bound — see §3.1b): the answer is produced from a
   *perfect textual description* of the GT visual content, bypassing the VLM's eyes entirely.

Metric: of the free-wrong cases, how many become correct at each rung.

- **“Fundamental wall”** = still wrong even at the strongest *visual* aid (rung 3/4).
- **“Perception-limited”** = recovered by better perception (rungs 2–4). This slice is, by
  construction, type-2 (frame/region selection) — i.e. the agent’s only real edge, and
  non-internalizable under your constraints.
- **“Reasoning-bound”** = still wrong at rung 4 but CORRECT at rung 5 (oracle perception).
  This is the slice where internalizing reasoning could help — its size per cell is what
  places that cell in regime 2 of the map.

### §3.1b ORACLE-perception upper bound (cheapest, most direct proof of "perception is the wall")

The §4 sighted-critic still routes through a VLM's eyes. The cleanest control removes vision
from the answer path entirely: feed a **ground-truth textual description** of the scene to the
**text reasoner** (the 30B) and let it reason to the answer with NO image access.

- Source of the GT description, per dataset: NExT-GQA — the human caption / action annotation
  for the GT window (or a strong-VLM caption of the GT-localized hi-res frames, audited);
  ChartQA — the underlying data table (ChartQA ships the table); CLEVR(ER) — the scene graph /
  program (synthetic GT is exact). Document the oracle source per dataset; flag any that use a
  strong-VLM caption as "near-oracle" not "oracle."
- Read it as: **oracle accuracy = the reasoning ceiling once perception is free.** If
  oracle ≈ free_form → reasoning isn't the bottleneck (perception-bound, regime 1). If
  oracle ≫ free_form → reasoning is reachable but the VLM's perception is the wall
  (perception-bound, but with a clear reasoning ceiling above); if oracle is high AND the VLM
  free_form already perceives well (rung-3 recovers most), the residual is reasoning-bound
  (regime 2). This single number disambiguates the regimes per cell.

### §3.2 Run the probe on every dataset used as a main result

NExT-GQA (have it), plus whichever §5 datasets are promoted to “main.” Report a single
cross-dataset table: dataset → % perception-limited vs % fundamental.

### §3.3 Tie mechanism to the headroom result → place each cell on the MAP

For each (dataset, base) cell, you now have: net (§1, on EVIDENCE_IN ∩ CLEAN), %perception-
limited, %fundamental, %reasoning-bound, and oracle accuracy (§3.1b). Use them to place the
cell:
- **Regime 1 (perception-/selection-bound):** net within variance AND difficulty is
  perception/selection (rung-3 recovers much, or oracle ≫ free_form). → no internalizable
  reasoning headroom. (Expected: NExT-GQA all scales; CLEVRER all scales; ChartQA at 4B/8B.)
- **Regime 2 (reasoning-bound):** free_form already perceives well (rung-3 adds little) AND a
  nonzero reasoning-bound slice remains (oracle high, residual is multi-step reasoning). →
  internalization has headroom; candidate for the §11 PoC. (Expected: ChartQA at 32B.)

The unified statement is the MAP, not a uniform negative: *"frames-visible difficulty is
perception/selection-bound and shows no internalizable reasoning headroom; the reasoning-bound
regime appears only once a strong-enough base has solved perception (e.g. 32B on static
charts), and there internalization is feasible."*

**Acceptance:**

- [ ] one perception-probe script, dataset-agnostic, config per dataset for GT localization
  AND oracle source.
- [ ] cross-dataset table: dataset × base → %perception-limited, %fundamental, %reasoning-bound,
  oracle acc, regime {1|2}.
- [ ] explicit linkage: net-verdict (§1) co-classifies with regime; ≥1 cell in each regime.

-----

## §4. Critical missing control — sighted critic ablation (closes the mechanism loop)

**Purpose:** your mechanism story is “orchestrated reflection fails because the 30B critic
is blind and integrates the small VLM’s perceptual garbage.” A reviewer will immediately
ask: is it that the critic is *blind*, or that the small VLM’s perception is *unfixable*?
You currently cannot answer this. Add the sighted-critic condition.

### §4.1 Build `orch_reflect_sighted`

Same orchestrated-reflection loop, but the critic is a VLM that SEES the same fixed
uniform-N frames (e.g. the 32B-dense VLM you already have, or the strongest VLM you can
serve). It re-examines and integrates, with vision.

### §4.2 Three-way comparison on the main slice

free_form vs orch_reflect_blind vs orch_reflect_sighted, multi-seed (§1 protocol).

- If sighted critic ALSO fails to beat free_form → the wall is the small VLM’s frames /
  perception ceiling itself (strongest version of your claim).
- If sighted critic DOES help → the wall was critic-blindness, which reframes (but doesn’t
  kill) the result: “agentic reasoning needs a sighted reasoner, blind text orchestration is
  net-negative.” Either outcome is publishable; you must run it to know which.

**Acceptance:** [ ] three-way table with CIs on ≥ 2 datasets; explicit verdict on
blind-vs-sighted as the locus of failure.

-----

## §5. Dataset coverage — depth-matched, not breadth-padded

**Purpose:** the log has uneven depth (NExT-GQA full suite; CLEVRER/ChartQA only orch-gap).
Reviewers see this instantly. Fix by choosing a small number of MAIN datasets done in full,
and demoting the rest to clearly-labeled robustness case studies.

**Choose the two main datasets to SPAN the map's two regimes** (not "two perception-bound
sets") — that is what makes the contribution a map rather than a one-sided negative.

### §5.1 Designate datasets

- **Main A — the perception-bound regime (video):** NExT-GQA causal/temporal CLEAN ∩
  EVIDENCE_IN (have it). Full suite, all scales (4B/8B/32B), multi-seed. Expected: regime 1,
  net within variance, perception/selection-bound. This is the boundary half of the map.
- **Main B — the regime-crossing dataset (perception → reasoning with scale):** a
  perception-clean STATIC reasoning set where the base scale flips the bottleneck. **ChartQA
  is the exemplar from Exp-8** (4B/8B perception-bound ~.64 → 32B perception solved .80,
  residual = multi-step arithmetic = regime 2). Add **MathVista (or GQA)** if you want a
  second, harder reasoning-bound point. Full suite, all scales, multi-seed. This is the
  *positive* half — it is NOT a "perception-bound control"; it is where reasoning headroom
  appears and where the §11 PoC runs.
- **Robustness case study (orch-gap + perception probe only, multi-seed):** CLEVRER — the
  clean illustration that "rendered ≠ perceivable" (motion-tracking is perception). Explicitly
  secondary; shows the mechanism (perception amplification under forced sub-reads).

> Correction to the original draft: ChartQA was filed as a perception-bound control. Exp-8
> shows it is the *regime-crossing* dataset and belongs in MAIN (it carries the positive
> half of the map). Do not demote it.

### §5.2 Per-dataset GT localization config

Each dataset needs a way to produce “GT-localized frames/region” for §3. Document per
dataset what the GT is (temporal grounding, bbox, etc.) and the localization function.

**Acceptance:**

- [ ] 2 main datasets, identical full suite, multi-seed, with CIs.
- [ ] 2 robustness datasets, orch-gap + perception probe, clearly labeled secondary.
- [ ] no dataset presented as main evidence without the full suite.

-----

## §6. Cross-family generalization — break the Qwen monoculture (targeted, not exhaustive)

**Purpose:** all current models are Qwen3-VL. A reviewer will ask whether “perception-bound”
is a property of small VLMs generally or of Qwen specifically. Answer it with ONE additional
family, minimally.

### §6.1 Add 1 (ideally 2) non-Qwen families

- Primary add: a 7–8B VLM from a different lineage / vision encoder — **InternVL3-8B**
  preferred (different vision tower + training data), or LLaVA-OneVision-7B.
- **Pick for a DIFFERENT vision-encoder paradigm**, not just a different brand: Qwen3-VL uses
  dynamic-resolution tiling; a model with a fixed-resolution ViT (InternVL / LLaVA-OV) tests
  whether "perception-bound on video" is a property of Qwen's tiling or of small VLMs broadly.
  That is the actual scientific question §6 answers.
- **Serving caveat (learned the hard way):** InternVL AWQ on ModelScope is lmdeploy-format
  (`quant_method` null in config) → **may not load in vLLM**. Verify load before committing;
  fall back to bf16 (InternVL3-8B ≈ 16 GB, fits 1 card) or a vLLM-native quant. Budget time
  for this; it bit the 32B-AWQ run (masked_scatter at mem-util 0.90 → use 0.85).
- Optional second: Pixtral-12B or Gemma-3-VL, only if serving budget allows.
- Constraint: must be servable on your hardware for inference (no training needed for the
  diagnostics; the §11 PoC is the only part that trains, and only on the regime-2 base).

### §6.2 Minimal cross-family protocol (do NOT replicate the full grid)

On the MAIN NExT-GQA slice only, run for each new family:
free_form, orch_reflect_blind, perception probe — multi-seed.

- Purpose is a single cross-family verification point: does net ≈ 0 and perception-bound
  hold off-Qwen? If yes, your claim generalizes beyond one family. If a family behaves
  differently, that itself is a finding (and you report it).

**Acceptance:** [ ] ≥ 1 non-Qwen family run on the main slice with the minimal protocol,
multi-seed; cross-family row added to the headline table.

-----

## §7. Scale-up & infra hardening (enabling work, parallelizable)

**Purpose:** §1–§2 need n ≥ 200–300 per slice; you ingested only 70–120. Fix the ingest
bottleneck and scale.

- **GPU-accelerated ingest:** resolve the SigLIP/BGE fp16-alongside-vLLM segfault (deferred
  in the log). Either isolate ingest on a dedicated GPU/process, or run ingest in a separate
  env with no vLLM context loaded. Target: ingest the NExT-GQA causal subset to ≥ 300 cases.
- **Seed-loop harness:** wrap every `diag_*` script so it takes `--seeds 0..9` and emits
  per-seed JSON + an aggregated CI report. This is the single most reused piece of new code;
  build it once in `app/distill/seed_runner.py`.
- **Result store:** one tidy results table (parquet/jsonl) keyed by full fingerprint, so
  the paper tables regenerate from a single query. No hand-copied numbers.

**Acceptance:**

- [ ] ingest scaled to ≥ 300 cases on the main slice, 0 errors.
- [ ] `seed_runner` produces aggregated CI reports for any diag script.
- [ ] all paper numbers regenerate from the result store via one script.

-----

## §8. Reporting artifacts (what the paper/PI-facing writeup needs)

Build these as the experiments land, not at the end.

1. **THE MAP (headline figure):** a (dataset × base) grid, each cell colored by regime
   {1 perception/selection-bound | 2 reasoning-bound}, annotated with net±CI and oracle acc.
   This single figure IS the contribution; build everything to fill it.
1. **Master results table:** rows = settings (dataset × model × method), cols = free_form
   acc (CLEAN ∩ EVIDENCE_IN), net_mean, 95% CI, verdict {effect|within-variance}. Multi-seed.
1. **Mechanism table:** dataset × model → %perception-limited, %fundamental, %reasoning-bound,
   oracle acc, regime.
1. **The retraction box:** “initial 8B +8% → within variance after multi-seed + clean-label” —
   methodological contribution about VLM-reasoning eval variance AND label noise.
1. **Label-noise box (§0.5):** label-noise rate per dataset + how clean-label changed σ/power.
1. **Blind-vs-sighted-vs-oracle row:** §4 + §3.1b — perception ceiling vs reasoning ceiling.
1. **Cross-family row:** §6.
1. **(If §11 run) PoC result:** internalized-reasoning Δ on the regime-2 cell, with CI.
1. **Power statement:** §2, one sentence with the detectable-effect number on CLEAN.

-----

## §9. Execution order (gate-respecting; for CC scheduling)

1. §0 conventions + §0.5 **text-grader fix + label audit / clean-label** + §7 seed_runner +
   result store                                     (infra + the floor under the floor)
1. §3.0 **frames-visible partition** (EVIDENCE_IN/OUT) on the main slice  (gates §1's claim)
1. §1 variance gate on NExT-GQA 8B & 4B, on CLEAN ∩ EVIDENCE_IN  (THE gate; greedy + bootstrap)
1. §2 power analysis on CLEAN                        (cheap, right after σ known)
1. §3.1b **oracle-perception upper bound** + §4 sighted-critic, NExT-GQA  (closes mechanism)
1. §3 standardized perception probe, cross-dataset → **place cells on the MAP**
1. §5 main A (NExT, regime 1) + main B (ChartQA, regime-crossing) full suite, all scales
1. §6 one non-Qwen family (different vision encoder), minimal protocol  (generality)
1. **§11 PoC: internalize reasoning in the regime-2 cell** (high-upside; positive half)
1. §5 robustness (CLEVRER) + §8 final tables incl. THE MAP                (polish)

Stop-and-reassess after step 4: §0.5+§3.0+§1 decide, per cell, regime 1 vs 2. Confirm at
least one regime-2 cell exists (Exp-8 says 32B-ChartQA does) BEFORE deciding whether to invest
in §11. If §1 collapses everything to regime 1 even on ChartQA-32B-CLEAN, fall back to the
pure boundary result (still publishable); if regime 2 survives clean-label, §11 is the payoff.

-----

## §10. Scope discipline (read when tempted to expand)

This is an RA/PhD-application-grade decomposition+map + methodology note, not a NeurIPS-main
sweep. **Minimum publishable/sharable core = §0.5 (clean labels) + §3.0 (frames-visible) + §1
(variance gate) + §2 (power) + §3 incl. §3.1b oracle (NExT) + §4 (sighted critic) + the
retraction box + the 2-cell map (NExT regime-1, ChartQA-32B regime-2).** This core is a
complete, defensible story by itself.

Additive strength, in priority order, dropped if time runs out: §5 main B full suite across
scales, §6 cross-family, §11 PoC, §5 robustness datasets. **§11 (the SFT PoC) is the single
highest-upside add** — it converts "negative boundary + map" into "boundary + map + a working
positive internalization," which is the difference between an RA-grade and a PhD-grade artifact.
Do it if the regime-2 cell survives clean-label (§9 step-4 checkpoint).

Do NOT open a new dataset or model beyond this spec without first asking whether it defends the
central contribution (the map), not just adds a number.

-----

## §11. (High-upside, optional) Constructive PoC — internalize reasoning in the regime-2 cell

**Purpose:** close the loop the whole original project was about. In a cell the map places in
regime 2 (perception solved, residual = reasoning — Exp-8 candidate: **32B on ChartQA /
MathVista**), show that the agentic/orchestrated reasoning CAN be internalized into a single
forward pass — i.e. an actual positive internalization result, with the negative boundary
(regime 1) as the foil. This is what makes the work a PhD-grade *contribution* rather than only
a *diagnosis*.

### §11.1 Generate + filter CoT in the regime-2 cell
- Run `orch_reflect_sighted` (or the strongest reasoner you have) on the regime-2 training
  split to produce step-by-step rationales that reach the correct answer; keep only
  reachable+correct CoTs (reuse the consistency-gate idea: keep CoTs the *target* base can act
  on, answer stripped — `app/distill/filter_consistency.py`). Target ~1–3k clean CoTs.
- The teacher here MAY be a stronger reasoner (the project's self-improvement constraint was
  already retired in the 2026-06-08 pivot; this is orchestrator/teacher distillation, stated
  honestly).

### §11.2 SFT the SMALL base in that cell, evaluate single-forward
- LoRA-SFT the small VLM (the base that was perception-OK but reasoning-weak in that cell —
  if 32B solved perception, the interesting target is "does an 8B that *can read charts* but
  *can't do the arithmetic* improve?"; pick the smallest base whose rung-3 perception is
  adequate). Train on (image, question, CoT, answer).
- Evaluate **tool-free single forward** vs the pre-SFT base on a held-out, video/scene-isolated
  split, multi-seed, on CLEAN. Headline = Δacc with CI.

### §11.3 Causal probe (lightweight — the one that matters)
- 2a counterfactual: corrupt one intermediate numeric conclusion in the model's own CoT, force
  continuation, measure answer flip-rate vs a shuffled-CoT control. Real reasoning flips;
  template does not. This is the single most convincing probe and is cheap.

**Acceptance:**
- [ ] regime-2 cell confirmed (§9 step-4) before any training.
- [ ] internalized single-forward Δacc with CI > 0 on CLEAN held-out, OR an honest null with
  the power statement (either is reportable — a null here just shrinks the map's regime-2).
- [ ] 2a flip-rate vs shuffled-CoT control, showing the gain is reasoning not memorization.

**Scope guard:** this is the ONLY section that trains. If serving/time is tight, the core
(§10) stands without it; but it is the highest-leverage hour for a PhD application.