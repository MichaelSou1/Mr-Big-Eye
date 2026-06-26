# Full-SFT 8B Non-Video Audit

Overall: COMPLETE

## Requirement Status

| requirement | status |
|---|---|
| training_all_arms | PASS |
| source_lora_8b_arms_found | PASS |
| lora_to_full_sft_8b_coverage | PASS |
| chartqa_eval | PASS |
| chartqa_full_battery | PASS |
| tabmwp_eval | PASS |
| tabmwp_full_battery | PASS |
| tabmwp_full_battery_has_answers | PASS |
| tabmwp_posthoc_ready | PASS |
| finqa_targeted_all_arms | PASS |
| tabmwp_weight_retained_until_battery_done | PASS |
| tabmwp_weight_clean_after_battery_done | PASS |

## LoRA-to-Full-SFT 8B Coverage

| arm | source LoRA train | source LoRA eval | source LoRA probe | Full-SFT train | Full-SFT eval | Full-SFT probe | Full-SFT coverage |
|---|---|---|---|---|---|---|---|
| chartqa | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tabmwp | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| finqa_b2 | PASS | NA | PASS | PASS | NA | PASS | PASS |
| finqa_vanilla | PASS | NA | PASS | PASS | NA | PASS | PASS |
| finqa_b2_text | PASS | NA | PASS | PASS | NA | PASS | PASS |
| finqa_vanilla_text | PASS | NA | PASS | PASS | NA | PASS | PASS |

## Training Arms

| arm | status | data | usable | epochs | trainable | frozen early |
|---|---|---|---:|---:|---:|---:|
| chartqa | PASS | data/distill/poc/chartqa_cot_train.jsonl | 150 | 2.0 | 79.7% | 3 |
| tabmwp | PASS | data/distill/poc/tabmwp_cot_train.jsonl | 160 | 2.0 | 79.7% | 3 |
| finqa_b2 | PASS | data/distill/finqa/curriculum_dev_strict.jsonl | 180 | 3.0 | NA | 3 |
| finqa_vanilla | PASS | data/distill/finqa/curriculum_dev_none.jsonl | 180 | 3.0 | NA | 3 |
| finqa_b2_text | PASS | data/distill/finqa/curriculum_dev_strict_text.jsonl | 173 | 3.0 | NA | 3 |
| finqa_vanilla_text | PASS | data/distill/finqa/curriculum_dev_none_text.jsonl | 173 | 3.0 | NA | 3 |

## Full-SFT Eval

| dataset | status | n_eval | base_acc | full_acc | net | gain/lost | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|
| chartqa | PASS | 400 | 71.2% | 76.2% | 5.0% | 39/19 | 0.0126020249399363 |
| tabmwp | PASS | 400 | 86.5% | 96.8% | 10.3% | 42/1 | 1.0610641964594883e-09 |

## Chart/Table Batteries

| cell | status | n_eval | base_acc | interventions | per-case answers | answer variants | missing answer variants | corrupt_flip | shuffle_flip | para_flip | snap | follow |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|
| chartqa_present | PASS | 305 | 76.2% | corrupt,delete,filler,paraphrase,shuffle,truncate | NO | NA | NA | 10.0% | 25.9% | 4.9% | 91.7% | 3.0% |
| chartqa_masked | PASS | 305 | 76.2% | corrupt,delete,filler,paraphrase,shuffle,truncate | NO | NA | NA | 24.9% | 38.0% | 9.5% | NA | NA |
| tabmwp_present | PASS | 387 | 96.8% | corrupt,delete,filler,paraphrase,shuffle,truncate | YES | corrupt,delete,filler,paraphrase,shuffle,truncate | NA | 5.2% | 15.5% | 7.2% | 97.2% | 1.6% |
| tabmwp_masked | PASS | 387 | 96.8% | corrupt,delete,filler,paraphrase,shuffle,truncate | YES | corrupt,delete,filler,paraphrase,shuffle,truncate | NA | 22.7% | 29.5% | 8.3% | NA | NA |
| tabmwp_present_core | PASS | 385 | 96.2% | corrupt,shuffle | NO | NA | NA | 4.9% | 16.1% | NA | 97.1% | 1.6% |
| tabmwp_masked_core | PASS | 385 | 96.2% | corrupt,shuffle | NO | NA | NA | 25.5% | 29.9% | NA | NA | NA |

## FinQA Targeted

| arm | status | n_eval | n_targeted | base_acc | operand_follow | consistent_follow | shuffle_snap |
|---|---|---:|---:|---:|---:|---:|---:|
| finqa_b2 | PASS | 169 | 165 | 64.8% | 0.0% | 98.2% | 63.0% |
| finqa_vanilla | PASS | 157 | 155 | 60.2% | 0.0% | 98.1% | 71.0% |
| finqa_b2_text | PASS | 167 | 161 | 64.0% | 0.0% | 97.5% | 81.4% |
| finqa_vanilla_text | PASS | 156 | 152 | 59.8% | 0.7% | 94.7% | 82.9% |

## TabMWP Eval

- base -> full: 86.5% -> 96.8%
- net: 10.3%; gain/lost: 42/1; McNemar p: 1.0610641964594883e-09

## TabMWP Posthoc

- status: PASS
- path: `data/distill/poc/full8b_tabmwp_battery_posthoc.json`
- cells: {'tabmwp_present': 'PASS', 'tabmwp_masked': 'PASS'}

## Resume Readiness

| item | current | expected/notes |
|---|---:|---|
| tabmwp_mimo_paraphrase | 387 | 387 after present run; data/distill/poc/paraphrase_cache_full8b_tabmwp_mimo.jsonl |
| tabmwp_base_cot | 400 | 400 after present run; data/distill/poc/paraphrase_cache_full8b_tabmwp_mimo_base_cot.jsonl |
| tabmwp_weight | absent | None GB; retain until full battery is complete; then remove large weight shards |

## Remaining Work

- None.
