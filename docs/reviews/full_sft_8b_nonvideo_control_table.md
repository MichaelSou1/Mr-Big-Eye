# Qwen3-VL-8B Dense/Full-SFT Non-Video Control Evidence

Dense/full-SFT here means the vision tower is frozen, embeddings are frozen,
and the first 3 language layers are frozen; ChartQA/TabMWP summaries report
about 79.7% trainable parameters.

## LoRA-to-Full-SFT 8B Coverage

| arm | source LoRA train | source LoRA eval | source LoRA probe | Full-SFT train | Full-SFT eval | Full-SFT probe | Full-SFT coverage |
|---|---|---|---|---|---|---|---|
| chartqa | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tabmwp | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| finqa_b2 | PASS | NA | PASS | PASS | NA | PASS | PASS |
| finqa_vanilla | PASS | NA | PASS | PASS | NA | PASS | PASS |
| finqa_b2_text | PASS | NA | PASS | PASS | NA | PASS | PASS |
| finqa_vanilla_text | PASS | NA | PASS | PASS | NA | PASS | PASS |

## SFT Accuracy Control

| dataset/arm | base acc | full-SFT acc | net | gain/lost | McNemar p | status |
|---|---:|---:|---:|---:|---:|---|
| chartqa | 71.2% | 76.2% | 5.0% | 39/19 | 0.0126020249399363 | PASS |
| tabmwp | 86.5% | 96.8% | 10.3% | 42/1 | 1.0610641964594883e-09 | PASS |

## Chart/Table Causal Battery

| cell | status | n_eval | base/free acc | corrupt flip | shuffle flip | F=corrupt-shuffle | paraphrase flip | snap | follow | interventions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| chartqa_present | PASS | 305 | 76.2% | 10.0% | 25.9% | -15.9% | 4.9% | 91.7% | 3.0% | corrupt, delete, filler, paraphrase, shuffle, truncate |
| chartqa_masked | PASS | 305 | 76.2% | 24.9% | 38.0% | -13.1% | 9.5% | NA | NA | corrupt, delete, filler, paraphrase, shuffle, truncate |
| tabmwp_present_core | PASS | 385 | 96.2% | 4.9% | 16.1% | -11.2% | NA | 97.1% | 1.6% | corrupt, shuffle |
| tabmwp_masked_core | PASS | 385 | 96.2% | 25.5% | 29.9% | -4.4% | NA | NA | NA | corrupt, shuffle |
| tabmwp_present | PASS | 387 | 96.8% | 5.2% | 15.5% | -10.3% | 7.2% | 97.2% | 1.6% | corrupt, delete, filler, paraphrase, shuffle, truncate |
| tabmwp_masked | PASS | 387 | 96.8% | 22.7% | 29.5% | -6.7% | 8.3% | NA | NA | corrupt, delete, filler, paraphrase, shuffle, truncate |

## FinQA Targeted Causal Probe

| arm | status | n_eval | n_targeted | base acc | operand follow | operand snap | consistent follow | consistent snap | shuffle snap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| finqa_b2 | PASS | 169 | 165 | 64.8% | 0.0% | 100.0% | 98.2% | 0.6% | 63.0% |
| finqa_vanilla | PASS | 157 | 155 | 60.2% | 0.0% | 100.0% | 98.1% | 1.9% | 71.0% |
| finqa_b2_text | PASS | 167 | 161 | 64.0% | 0.0% | 100.0% | 97.5% | 1.9% | 81.4% |
| finqa_vanilla_text | PASS | 156 | 152 | 59.8% | 0.7% | 98.7% | 94.7% | 2.6% | 82.9% |

## TabMWP Posthoc Answer Classification

- status: PASS
- path: `data/distill/poc/full8b_tabmwp_battery_posthoc.json`
- cells: {'tabmwp_present': 'PASS', 'tabmwp_masked': 'PASS'}


## Current Interpretation

- ChartQA full battery, TabMWP full six-intervention batteries, and FinQA targeted probes all pass the strict Full-SFT 8B control audit.
- TabMWP present/masked outputs include per-case `details[].answers`; the posthoc shuffle/filler/paraphrase answer classification is ready.
- The non-video Full-SFT control can be claimed complete under `scripts/audit_full_sft_8b_nonvideo.py --strict`.
