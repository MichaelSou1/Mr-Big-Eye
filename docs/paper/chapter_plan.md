# Chapter Plan — *Reading, not Reasoning*

> AAAI 投稿向（abstract 7/21、full 7/28）。本计划由 ars-plan（academic-paper plan 模式）Socratic 产出。
> 配套：[progress.md](../progress.md)（实验事实源）、[todo/0622.md](../../todo/0622.md)（执行清单 + WU 对账）。
> 日期：2026-06-23。

## 主轴句（contribution，A 式）
**把一个 agentic 视觉推理器蒸馏成单次前向，迁移的是“读”、不是承重的“算”——首个对这类模型的因果 CoT-忠实度审计证明它，并发现承重 locus 是 regime 依赖的。**

## Posture / Scope（锁定）
- **脊柱 = 硬化的因果审计/边界结果**（全在静态图表/表格，绕开视频 ingest segfault）。
- **建设性 = (B2) 课程式忠实度修复**（预注册，两向都 informative），用自家因果探针验收。
- **拔高版组织性发现 = 承重 locus 的 regime 依赖性**：图表/表格上感知重读承重、CoT 装饰；自然图（小 V* 探针）上文本 CoT 承重、感知被忽略（2510.23482 的镜像）；perception-bottleneck 预测哪极。
- **视频结果 → appendix**（边界动机：视频域 perception/选帧-bound、无可探的链）。
- **LOCUS（2606.16586）= 并发工作**（6/15，未索引），定位为“把*选帧*内化但只看 accuracy+attention，未做因果验证”。

## 三条 contribution bullet（差异点分散，不堆定语）
1. **发现**：蒸馏 agentic 推理 → reading 迁移、非承重链；模型重读图表；承重 locus regime 依赖。
2. **工具**：corrupt-vs-shuffle + **重读控制** + image-mask，**首次**用于蒸馏单遍 VLM。
3. **建设性 + 边界**：(B2) 预注册检验能否装出承重链；跨 regime 边界（视频进 appendix）。

---

## 章节骨架（~6.5k 正文 + appendix）

### §1 Introduction — ~900
- **目的**：agentic 视觉推理 + 内化进单遍 → 抛反直觉发现。
- **核心论点**：内化的不是过程、是感知/读；承重的是文本链还是感知随 regime 翻转。
- **证据锚**：8B ChartQA SFT .617→.733（CI 排除 0）但探针 corrupt≈shuffle≈低 → 读不是算。
- **Reviewer 风险**：“又一篇 faithfulness” → 主句=regime 依赖 + 重读机制 + 蒸馏单遍对象，三者皆不在现有工作。
- **强度**：Strong。

### §2 Related Work — ~700
- **核心论点**：内化工作只看 accuracy（VPD/Zooming-without-Zooming/PEARL/ChartPaLI/Chart-R1）；因果审计要么打工具型 MCoT（2510.23482，视觉被忽略）、要么打 latent（2602.22766，退化）、要么打工具输出（CodeV）——没人打蒸馏离散 CoT，没人发现 regime 极性翻转。
- **证据**：四列对比表（对象/干预/极性/机制全不同）。
- **Reviewer 风险**：撞 2602.22766（已发表 ICML 2026）/2510.23482 → 用对象+机制+控制三重区分措辞。
- **强度**：Strong（前提：全文已读、措辞精确）。

### §3 Setup — ~700
- **核心论点**：teacher CoT → consistency 过滤 → SFT 进单遍学生；评测用 ATE-on-accuracy + McNemar（沿用 2510.23482 协议，降摩擦）。
- **证据**：§11/§7.13 现成管线；ChartQA(150 CoT)+TabMWP；train/test disjoint。
- **Reviewer 风险**：“蒸馏 chart CoT 是旧的” → 那是底物不是 claim。
- **强度**：Moderate→Strong。

### §4 The Causal Faithfulness Audit（方法核心）— ~900
- **核心论点**：corrupt（改中间值）vs shuffle（打乱序）成对；image-mask（去图留 CoT，逼出潜在链）；**re-perception 重读控制**（corrupt 后答案吸附真值还是注入错值 → 直证读不是算）。
- **证据**：现成 `poc_causal_probe_32b.py`；重读控制为新 named control。
- **Reviewer 风险**：2502.14829“删步只测 contextual faithfulness” → 重读控制测答案更新行为，正面绕开。
- **强度**：Strong。

### §5 Findings: Reading, not Reasoning + Regime 依赖（心脏）— ~1300
- **5.1 图表/表格极**：corrupt≈shuffle≈2%（32B 图在场）→ CoT 装饰、重读；image-mask 下 corrupt 10.9%>shuffle 6.5% → 链**学到但被绕过**；base 越强越 bypass（独立复现 Lanham）。
- **5.2 自然图极（N3 新探针）**：V* 子集文本 CoT 承重、感知被忽略——2510.23482 镜像。
- **5.3 Regime 轴**：perception-bottleneck（答案可否从图直接重读）预测哪极承重。
- **Reviewer 风险**：“V* 探针太小” → 定位为佐证双极、主证据在图表、n 由 WU-1 撑。
- **强度**：Strong（拔高点）。

### §6 Constructive: Can a Load-Bearing Chain be Installed?（B2）— ~700
- **核心论点**：(B2) 课程（只用“重读单格得不出、必须多步”的题）训练，预注册检验 corrupt-flip 能否显著>vanilla-SFT。
- **证据**：现成 SFT 管线 + 探针；prereg 两向都成立。
- **Reviewer 风险**：撞 SCCM(2510.23482)/CapImagine(2602.22766) → 他们修视觉/换 latent，你修文本链、机制不同。
- **强度**：Moderate（net-new，预注册保两向）。

### §7 Conclusion + Limitations — ~400
- 承重 locus regime 依赖；蒸馏迁移读不迁移算；(B2) 两向结果；限制：n、域（图表/表格+小 V*）、written-CoT 定义。

### Appendix
- 视频边界（NExT/CLEVRER/Penguin 4 scale × 3 encoder 图）；扩展干预（truncate/delete/paraphrase/filler）；power 表；(B2) 跨-teacher 备选臂（旧 WU-4.1）。

---

## INSIGHT Collection
- **thesis**：主轴句 A（reading-not-reasoning + regime 依赖）。
- **scope**：图表+表格主体（ChartQA+TabMWP）；小 V* 双极探针；视频→appendix。
- **contribution_claim**：A + 3-bullet 分布。
- **organizing_finding**：承重 locus 的 regime 依赖性（拔高版核心）。
- **novelty_position**：对象=蒸馏单遍离散 CoT；vs 2510.23482（工具 MCoT/视觉被忽略）、2602.22766（latent/退化）、CodeV（工具输出）、Lanham（文本 LLM）。
- **prereg**：若 (B2) corrupt-flip 未显著>vanilla-SFT，则感知捷径被深度偏好。
- **redlines**：① 不 claim corrupt 干预本身（Lanham/2510.23482 已有）；② abstract 不用“内化推理不承重”当主句（2602.22766 已发表占有）。

## Related Work 必引清单
- **方法谱系**：Lanham 2307.13702、Turpin 2305.04388、Pfau 2404.15758；批评 2502.14829、2402.14897。
- **三篇威胁**：2510.23482、2602.22766、CodeV 2511.19661。
- **内化（只看 acc）**：VPD 2312.03052、Zooming-without-Zooming 2602.11858、PEARL 2604.08065、ChartPaLI 2403.12596、Chart-R1 2507.15509。
- **图表动机锚**：ChartMuseum 2505.13444、Perception Bottleneck 2503.18435、CharXiv 2406.18521、CHART-NOISe 2509.18425。
- **表格底物**：TabMWP/PromptPG 2209.14610。
