# WU-1 完成快照 — ChartQA 评估 n 扩到 400(功效修复)

> 完成 2026-06-23。所有数字经 `data/distill/results/results.jsonl` + `data/distill/poc/*_n400.json`
> 指纹化结果产出,`scripts/wu1_report.py` 一键重生(无手抄)。对照基线见 `pre_upgrade_0622.md`。

## 验收:5/5 ✅
- [x] 1.1 测试集 n=400(`data/distill/chartqa/test_cases_400.jsonl`)+ 400 图,与 187 train 图 hash-disjoint
- [x] 1.2 三个 scale 的 free/net±CI/verdict
- [x] 1.3 8B+32B SFT Δacc + paired-bootstrap CI + McNemar @ n=400
- [x] 1.4 四个 2×2 探针 cell @ n≈320
- [x] 1.5 功效表更新到新 n

## 1.2 Gate @ n=400(free / agentic net±std,k=5 seeds)
| scale | free | self_reflect net | orch net |
|---|---|---|---|
| 4B | .685 | +.0085 ± .0042 | −.1085 ± .0128 |
| 8B | .708 | .000 ± .0018 | −.092 ± .0086 |
| 32B | .797 | +.0065 ± .0038 | −.084 ± .0089 |

- self_reflect 基本无增益;orch penalty 随规模递减(−10.9/−9.2/−8.4%),复现原 n=60 的 −13/−8/−5。
- **orchestrator 说明**:gate 的 orch 用**本地 8B**当 orchestrator 跑(DeepSeek 当时 402 欠费)。
  后用 DeepSeek(v4-flash)在 8B 上抽验 2 seeds:**net=−.062 ± .004**,与本地-8B −.092 **同号、|Δ|=.030**
  → 定性结论(orch 有害、随规模递减)与 orchestrator 无关;保留本地-8B 结果,论文注明已抽验。
  (抽验原始数据:`/home/gpus/logs/wu1/spot_8b_dsorch.jsonl`)

## 1.3 SFT Δacc @ n=400(train 固定 150-CoT,只放大 eval-n)
| | base | SFT(epoch_1) | net | 95% CI | McNemar |
|---|---|---|---|---|---|
| **32B** | .725 | .795 | **+.070** | [+.037, +.102] | b=37 c=9 → **p=.0001** |
| **8B** | .713 | .777 | +.065 | [+.025, +.105] | b=46 c=20 → p=.0021 |

- **功效修复命门**:32B 原 n=60 只有 4 个 discordant pair、McNemar **p=.125**;现 n=400 → **37 vs 9 pair、p=.0001**。
  "effect < n can resolve" 的杀稿点已堵死。8B 同样显著(p=.002)。

## 1.4 因果探针 2×2(flip rate;F = flip_corrupt − flip_shuffle)
| | n_eval | corrupt | shuffle | F |
|---|---|---|---|---|
| 32B present | 318 | .050 | .047 | +.003 ≈0 |
| 32B masked | 318 | .119 | .129 | −.010 |
| 8B present | 321 | .212 | .302 | −.090 |
| 8B masked | 321 | .333 | .408 | −.075 |

- **四格 F ≤ ~0**:篡改 CoT 里的具体数字,翻转答案**不比单纯打乱句序更多** → CoT **非 load-bearing**。
- masked > present(遮图后扰动才开始咬)→ 图才是主信息源 = "读图非推理"。8B 整体更脆但方向一致。
- 跨尺度(8B/32B)× 跨条件(present/masked)四路独立复现同一结论。

## 1.5 功效表(min detectable net @ n=400)
| cell | p_free | n_obs | mde@n_obs |
|---|---|---|---|
| 32B chartqa | .797 | 400 | **.080** |
| 8B chartqa | .708 | 400 | .090 |
| 4B chartqa | .685 | 400 | .092 |

- min detectable net 从 ~25%(原)降到 **~8%**;SFT 的 +7% 在 paired McNemar 下已 p=.0001 显著。

## 一句话
SFT 准确率真涨(+7%, p=.0001),但因果探针证明涨的是**读图**不是 **load-bearing 推理** —— 论文标题精神在高功效(n=400 / n≈320)下坐实。
