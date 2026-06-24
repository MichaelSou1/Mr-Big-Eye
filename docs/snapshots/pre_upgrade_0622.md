# Pre-upgrade numbers snapshot — 2026-06-22

Frozen "before" numbers for the `todo/0622.md` investability upgrade, so post-upgrade
results can be compared 1:1. Sources: `data/distill/results/{results.jsonl,power_table.json}`
(md5 of results.jsonl at snapshot: `6ab537f036e22f6f82b9d23f8ad46c22`, 143 lines;
backed up to `results.bak.0622.jsonl`) and `docs/progress.md §7.11–7.13`.

## Evaluation n (the power soft-spot the upgrade targets)
- NExT-QA: **n = 47**
- ChartQA: **n = 60**

## ChartQA 32B-SFT PoC (§11.4, QLoRA, the regime-2 reasoning-bound cell)
- base (adapter-off, NF4, same CoT prompt): test_acc **.700**
- **peak = epoch 1: test_acc .767, gain 4 / lost 0, net +.067**
- paired-bootstrap 95% CI **[+.017, +.133]** (excludes 0)
- **McNemar exact p = 0.125** — only **4 discordant pairs** (all one-sided) → underpowered
- magnitude caveat: base free-form (no CoT prompt) = .80; SFT .767 has NOT exceeded it

## ChartQA 8B-SFT PoC (§11, LoRA, perception-bound cell)
- base-8B **.617 → SFT-8B .733**; paired-bootstrap net **+0.117, 95% CI [+0.050, +0.200]** (excludes 0)
- gain 7 / lost 0; **McNemar p = 0.023**

## Causal probe — 2×2 of {8B,32B} × {present,masked}, only **2 interventions** (corrupt / shuffle)
| base | condition | n | corrupt-flip | shuffle-flip | reading |
|---|---|---|---|---|---|
| 8B  | image present | 48 | 16.7% | 27.1% | shuffle ≥ corrupt → NOT load-bearing arithmetic |
| 8B  | image MASKED  | 48 | 37.5% | 27.1% | corrupt > shuffle → latent load-bearing |
| 32B | image present | 46 | 2.2%  | 2.2%  | both ≈0 → CoT ignored, re-reads chart |
| 32B | image MASKED  | 46 | 10.9% | 6.5%  | corrupt > shuffle → latent load-bearing |

## Power table (`power_table.json`, 80% power, α=.05)
- 4B-NExT: p_free .745, n_obs 47, **MDE@47 ≈ .252**; MDE@300 ≈ .100
- 8B-NExT: p_free .660, n_obs 47, **MDE@47 ≈ .274**; MDE@300 ≈ .108
- (ChartQA cells not yet in power table — WU-1.5 will add them at the scaled n)

## THE MAP regimes (§7.11) — coverage the upgrade widens
- Single dataset carrying the positive+causal half: **ChartQA** only
- Single teacher family: **Qwen3-VL** only
- ChartQA 32B = regime-2 (reasoning-bound, free .80, agentic +.017)
- ChartQA 8B-SFT = R1 base, +.117 from internalization

## Reviewer attack-surface this snapshot pins (what each WU must move)
1. **Power** — ChartQA n=60, 32B McNemar p=.125 (4 discordant pairs) → WU-1 scales n to 300+
2. **Single dataset** — only ChartQA on positive+causal side → WU-3 adds TabMWP
3. **Single teacher** — only Qwen3-VL → WU-4 adds DeepSeek-text + non-Qwen VLM teacher
4. **Single intervention pair** — only corrupt/shuffle → WU-2 adds truncate/delete/paraphrase/filler
5. **No headline figure** — all tables → WU-5 MAP heatmap + accuracy⊥faithfulness scatter
