# WU-5 完成快照 — faithfulness ⊥ accuracy 主轴 + 两张头条图

> 完成 2026-06-24。所有数字/图**全部由 result store 派生**(battery / SFT eval / map / tables),无手抄。
> 一键重生:`conda run -n mbe-up python scripts/build_faithfulness_axis.py`(**纯 CPU,不用卡**;
> 上游 GPU 产物 WU-1/WU-2 已落盘)。
> 底物 = ChartQA test n=400(present n_eval=318/321,masked 同量级)。

## 验收:6/6 ✅(todo §WU-5)
- [x] `F = flip_corrupt − flip_shuffle`(present)+ masked + masked−present gap + 二项 SE 定义
- [x] 与 SFT Δacc join → `data/distill/results/faithfulness.json`
- [x] 图 A:MAP 热图(regime 上色 + free/net±CI)→ `docs/paper/figures/fig_map.{pdf,png}`
- [x] 图 B:accuracy-gain × faithfulness 散点 → `docs/paper/figures/fig_decoupling.{pdf,png}`
- [x] master 表加 faithfulness 列 → `data/distill/results/faithfulness_master.md`
- [x] 两图全由 store 生成;已 `\includegraphics` 织入 `paper.tex`(此机无 pdflatex 未编译)

## 数据源(无手抄)
| 量 | 来源文件 |
|---|---|
| corrupt/shuffle flip-rate + 计数 + snap-rate | `data/distill/poc/battery_{8b,32b}_{present,masked}.json` |
| SFT base→SFT Δacc / CI / McNemar | `data/distill/poc/lora_{32b,8b}_chartqa/eval_n400.json` |
| regime / free_acc / best-agentic net | `data/distill/results/map.json` |
| best-agentic net 的 CI | `data/distill/results/tables.json`(master) |

## faithfulness.json 主结果(present,n_eval=318/321)
| cell | F=corrupt−shuffle | 95% CI | F_masked | gap(masked−present corrupt) | snap_rate | SFT Δacc (p) |
|---|---|---|---|---|---|---|
| **32b** | **+0.003** | ±0.034 | −0.009 | +0.070 | .981 | **+7.0% (p=6.9e-5)** |
| **8b** | −0.090 | ±0.067 | −0.075 | +0.121 | .816 | +6.5% (p=2.1e-3) |

- **解读**:SFT 把准确率显著推 +6.5~7.0%(McNemar p≤.002),但 present 下 F≈0(32B 落进噪声 band、8B 为负=corrupt 翻得比 shuffle *更少*)→ 准确率涨与 CoT 承重性**正交**。
- gap>0:遮图后 corrupt-flip 才抬头(+.07/+.12)→ 潜在链只在无图时浮现;有图时被重读绕过(snap .98/.82 直证)。

## 两张图(头条记忆点)
- **`fig_map.pdf`**(spec §8 首次渲染):dataset×model 网格,R2 暖色 / R1 冷色 / 无数据灰。
  **唯一 R2 = 32B-ChartQA**(感知已解、残差=多步算术);其余全 R1。每格标 free / net / 95%CI / regime。
- **`fig_decoupling.pdf`**(论文标题精神一眼图):x=ChartQA 准确率,y=F。
  - 8B/32B × {present 实心, masked 空心},F 带 95% 二项 CI;
  - 灰 band = `|F| < 95% 采样噪声`(= "CoT not load-bearing");
  - 每尺度一条 **SFT 准确率增益水平箭头**(base→SFT,标 Δacc 与 McNemar p)——
    **箭头沿 x 大幅右移(+7%),点却钉在 F≈0 band 上**:distillation 买的是 reading 不是 load-bearing CoT。
  - 诚实标注:F 测于 base 模型(箭头示意准确率轴移动而模型不离开 F≈0 带)。

## 产物清单
```
data/distill/results/faithfulness.json          # F × Δacc join,逐 cell(含 SE / gap / snap）
data/distill/results/faithfulness_master.md     # master(ChartQA)+ faithfulness 列
docs/paper/figures/fig_map.{pdf,png}            # 图 A
docs/paper/figures/fig_decoupling.{pdf,png}     # 图 B
docs/paper/figures/figures_snippet.tex          # 备用 \input 片段
docs/paper/paper.tex                            # 已织入两个 figure(§sec:map / §sec:probe)
```

## 一句话
扩 5 干预 + 重读控制(WU-2)之上,WU-5 把"准确率涨 vs CoT 承重性"**join 成一根可视主轴**:
SFT 准确率 +7% 显著、F 却钉在 0 —— `accuracy ⊥ faithfulness` 由 result store 一键出图,可直接进论文。
