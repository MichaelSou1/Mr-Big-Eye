# Master table — ChartQA, with faithfulness column (WU-5)

F = flip_corrupt − flip_shuffle (present); ≤0 ⇒ CoT not load-bearing. Δacc/p from SFT eval. All from the result store.

| dataset | model | best agentic net | verdict | SFT Δacc (p) | F (present) ± 95% | F (masked) | snap-rate |
|---|---|---|---|---|---|---|---|
| chartqa | 32b | +0.012 | within-variance | +7.0% (p<1e-3) | +0.003 ± 0.034 | -0.009 | 0.981 |
| chartqa | 4b | +0.018 | within-variance | — | — | — | — |
| chartqa | 8b | +0.070 | within-variance | +6.5% (p=0.002) | -0.090 ± 0.067 | -0.075 | 0.816 |
| chartqa | penguin2b | -0.078 | within-variance | — | — | — | — |
| chartqa | penguin8b | -0.006 | within-variance | — | — | — | — |
