# TabMWP Full-SFT Battery Posthoc

This CPU-only summary classifies forced-continuation answers from `details[].answers`.
Shuffle has no injected value by design, so the paper-facing quantity is snap-to-gold/other, not follow-under-shuffle.

| cell | status | n_eval | variant | n | snap | same as base | injected-follow | other |
|---|---|---:|---|---:|---:|---:|---:|---:|
| tabmwp_present | PENDING: missing file | NA | NA | NA | NA | NA | NA | NA |
| tabmwp_masked | PENDING: missing file | NA | NA | NA | NA | NA | NA | NA |

## Cell Status

- `tabmwp_present`: PENDING (missing file); source_sha256=NA
- `tabmwp_masked`: PENDING (missing file); source_sha256=NA

## Notes

- `same as base` is useful because base answers are correct for kept cases, but `snap` is the stricter paper-facing label.
- `injected-follow` is meaningful for corrupt only; for shuffle/filler/paraphrase it should be interpreted as a diagnostic collision with the corrupt injected value, not a planned intervention target.
