# TabMWP Full-SFT Battery Posthoc

This CPU-only summary classifies forced-continuation answers from `details[].answers`.
Shuffle has no injected value by design, so the paper-facing quantity is snap-to-gold/other, not follow-under-shuffle.

| cell | status | n_eval | variant | n | snap | same as base | injected-follow | other |
|---|---|---:|---|---:|---:|---:|---:|---:|
| tabmwp_present | PASS | 387 | shuffle | 387 | 336/387 (86.8%) | 327/387 (84.5%) | 0/387 (0.0%) | 51/387 (13.2%) |
| tabmwp_present | PASS | 387 | filler | 387 | 123/387 (31.8%) | 108/387 (27.9%) | 2/387 (0.5%) | 262/387 (67.7%) |
| tabmwp_present | PASS | 387 | paraphrase | 387 | 369/387 (95.3%) | 359/387 (92.8%) | 0/387 (0.0%) | 18/387 (4.7%) |
| tabmwp_present | PASS | 387 | corrupt | 387 | 376/387 (97.2%) | 367/387 (94.8%) | 6/387 (1.6%) | 5/387 (1.3%) |
| tabmwp_masked | PASS | 387 | shuffle | 387 | 283/387 (73.1%) | 273/387 (70.5%) | 4/387 (1.0%) | 100/387 (25.8%) |
| tabmwp_masked | PASS | 387 | filler | 387 | 27/387 (7.0%) | 27/387 (7.0%) | 0/387 (0.0%) | 360/387 (93.0%) |
| tabmwp_masked | PASS | 387 | paraphrase | 387 | 366/387 (94.6%) | 355/387 (91.7%) | 0/387 (0.0%) | 21/387 (5.4%) |
| tabmwp_masked | PASS | 387 | corrupt | 387 | 310/387 (80.1%) | 299/387 (77.3%) | 30/387 (7.8%) | 47/387 (12.1%) |

## Cell Status

- `tabmwp_present`: PASS (OK); source_sha256=572072b338c6
- `tabmwp_masked`: PASS (OK); source_sha256=a76a30fa6b20

## Notes

- `same as base` is useful because base answers are correct for kept cases, but `snap` is the stricter paper-facing label.
- `injected-follow` is meaningful for corrupt only; for shuffle/filler/paraphrase it should be interpreted as a diagnostic collision with the corrupt injected value, not a planned intervention target.
