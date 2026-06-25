# N1 完成快照 — 建设性 (B2):课程**装不出**承重链 → H_fail_bypass(预注册命中)+ 第三种 bypass「抄结论」

> 完成 2026-06-25。**H_fail_bypass 跨 2 尺度 × 2 teacher 类型四点共证**:8B/32B(gold-program teacher,§2/§8)
> + 文本推理器 teacher(§9,流畅 answer-conditioned 链,杀 CoT-格式 artifact)——四点 **operand-follow 全 0**,
> 抄结论同案签名 8B 165/165·32B 170/170·文本臂 155/156。base / vanilla-SFT / B2-SFT × operand/consistent/shuffle 三探针。
> **预注册先于结果**见 [n1_prereg_0625.md](n1_prereg_0625.md)(三向 H_success / H_fail_bypass / H_fail_fabricate)。
> 所有数字经 `data/distill/poc/battery_n1_8b.json` 指纹化产出;一键重生:
> 训练集 `python scripts/build_n1_curriculum.py --split dev --filter {strict,none} --teacher gold_program`;
> 测试集(probe 源)`--split test --filter strict`;SFT `bash scripts/run_n1_sft.sh`;
> 探针 `CUDA_VISIBLE_DEVICES=0,1 python scripts/battery_n1_targeted.py --base <8B> --quant none --adapters b2=… vanilla=… --dump data/distill/finqa/curriculum_test_strict.jsonl --out data/distill/poc/battery_n1_8b.json`。

## 一句话(预注册命中 H_fail_bypass + 一条新机制 + 一个方法学红利)
在 FinQA 渲染表(感知瓶颈:base free-form acc **.08**)上,用**最强可能的建设性配方**——课程只留
③ 多步·多格·有可翻操作数·非单格可读的题,teacher 用**确定性、可证承重的 gold-program→NL 链**——SFT 出的
学生链**仍非承重**:**定向 corrupt 一个经 gold-program 验证的承重输入操作数,follow=0/172(B2)、0/147(vanilla)、
0/21(base)**(全 Wilson CI 上界 ≤.025),snap≈1.0,B2≈vanilla≈base。课程把**准确率**从 .08→.582(vanilla)
→**.674(B2,+9.2 pt over vanilla)**真推高,却**完全没动承重性**——**accuracy ⊥ faithfulness 在第四个底物、用
训练期干预再次坐实**。机制:学生既不重读图(读不动),也不从操作数重算,而是**直接抄链尾写好的结论**
(consistent corrupt 把结论改错→follow .96;operand-only 留对结论→snap 1.0)——**reading-not-reasoning 的第三种
bypass:「读结论,不走步骤」**。

## 验收(对应 todo §N1)
- [x] 选数据集 + 预注册先行(FinQA gold-program 预言机)→ `n1_prereg_0625.md`(先于结果)。
- [x] 课程过滤脚本落成 `scripts/build_n1_curriculum.py`(执行器 + ③ 过滤 + ② 可翻操作数标注 + 表→图渲染 + 双 teacher 臂)。
- [x] **生成课程 + vanilla 训练集**:dev strict **180**(n_ops {2:141,3:18,4:8,5:13},全 ≥2 步)vs dev none **180**
      (n_ops {1:**95**,2:73,3:9,5:3},含 95 条单步捷径)——同一 gold-program teacher、同 n、独立图目录,**只差课程**。
- [x] **SFT 小 VLM(8B)**:B2(strict)+ vanilla(none)各一 LoRA,同配方(3 ep,lr 1e-4,r16),并行跑于错卡;
      train_loss b2 .215 / vanilla .357。
- [x] **定向 ② 探针** `scripts/battery_n1_targeted.py`:FinQA test 262 条 ③ 候选(261 有可翻表操作数),
      gold-program 预言机做 **operand-only**(改输入操作数,留下游结论)+ **consistent**(改操作数并一致传播到结论)
      两变体 + shuffle 控制,三分 snap/follow/other,base / vanilla / B2 一次载模。
- [x] 验收:**snap/follow/other(三模型 × 三探针)+ 二项 CI + 预注册三向判决** → **H_fail_bypass**(见 §2–§4)。

## 1. 操作化与改良(相对预注册的两点收紧,均更严)
- **teacher = gold_program(确定性、可证承重)做主报数臂**:gold-program→NL 是**构造上可证承重**的最强注入,
  若连「可证承重的 teacher 链」都装不出承重 student,边界结论更硬,且**消除 teacher 质量混淆**。课程对照(B2 vs
  vanilla)**同 teacher**,科学对比不受影响。**prereg ⑤ 原列文本推理器为主臂——已在 §9 补齐**(本地 orch 解 402
  见 [[deepseek-402-local-orchestrator]] + answer-conditioned 流畅 teacher),**逐结论复现**(operand-follow 0)→
  杀掉「结论只对 gold-program 那种死板 CoT 格式成立」之疑。
- **follow 口径细化(预注册 battery_followrate.py 口径的多步推广)**:多步题里「单操作数的注入值」几乎不可能等于最终答案,
  故 **follow := gold-program 用被改操作数重算出的新最终答案 `injected_final`**(不是操作数自身的值)——这正是选 FinQA
  「自带 gold program」的回报:能精确算出「若真走链」该落在哪。
- **两变体堵住 N3 冗余逃逸 + 抄结论逃逸**(新):operand-only 留对的结论(follow 需**真重算**,抄结论免疫);
  consistent 抹掉对的结论(堵 N3 式冗余)。两者夹出真相(见 §3)。

## 2. 头条数字(8B,FinQA test;`battery_n1_8b.json`)
| mode | base_acc | n_eval | n_tgt | **operand follow** | operand snap | consistent follow | shuffle other |
|---|---|---|---|---|---|---|---|
| base    | **.080** | 21  | 21  | **0/21 (.000, CI[.000,.155])**  | .952 | .857 | .095 |
| vanilla | **.582** | 152 | 147 | **0/147 (.000, CI[.000,.025])** | 1.000 | .980 | .361 |
| **B2**  | **.674** | 176 | 172 | **0/172 (.000, CI[.000,.022])** | 1.000 | .959 | .372 |

- **承重测试(operand-only)= follow 0 全员**:把一个**经 gold-program 验证会翻**的承重**输入**操作数改掉(2v+7),
  学生答案**一例都不跟随**重算值,而是照旧吐 gold(snap≈1.0、flip 0)。→ **链的「推理内容」(操作数/步骤)非承重。**
- **课程无效于承重**:B2 operand-follow(0/172)= vanilla(0/147)= base(0/21)。**课程 ≠ 承重杠杆**——预注册
  **H_fail_bypass** 命中(「follow≈0、snap 高、B2≈vanilla」逐字成立)。**H_success / H_fail_fabricate 均被否**
  (other 不升:operand other ≤ .048)。
- **课程**确实**升准确率**:.080(base)→ .582(vanilla)→ **.674(B2)**,B2 比 vanilla 高 **+9.2 pt**(课程 > 随机同量)。
  → **accuracy ⊥ faithfulness**:准确率动了 +59 pt,承重性钉死在 0。

## 3. 杀手级诊断:operand vs consistent 同案对照 → 机制是「抄结论」
同一批 case、同一 `injected_final`,只差「链尾结论是否被改成 injected」:
| 探针 | 链尾结论 | B2 follow | 解读 |
|---|---|---|---|
| **operand-only** | 仍是**对的 gold** | **0/172** | 改输入操作数 → 学生**不重算**,照抄对的结论 → snap 1.0 |
| **consistent** | 被改成 **injected** | **165/172 (.959)** | 结论被改错 → 学生**照抄错的结论** → follow .96 |

- **同案签名 165/165**:B2 中**每一个** consistent=follow 的 case,在 operand-only 下都是 snap(抄对结论)。
  → 那 96% 的「follow」**100% 是抄结论,0% 是重算**。学生**能**产出 injected_final(consistent 下产出了),
  但**仅当它被写进结论行**——**铁证:模型抄链尾结论,不从操作数走步骤。**
- **方法学红利**:**只看 consistent 会把 follow .96 误读成「承重」**;只有 gold-program 预言机驱动的 operand-only
  才揭穿 follow=0。→ **直接验证 N3 红线**「必须用冗余感知的定向探针」,并把它推进一步(还要堵「抄结论」逃逸)。
  这条 operand↔consistent 对照本身可复用、可写进论文当作「为何朴素整链 corrupt 会高估承重」的演示。
- **shuffle 控制**:follow≈0(.006/.014)、snap≈.62、other≈.37——打乱句序(gold 仍在文内)使「抄结论」失败 37%
  落 other,但**从不**跟随注入值 → 进一步确证驱动 follow 的是**被改的结论内容**,非任意编辑。

## 4. 预注册三向判决(先写、后看,逐条对账)
- **H_success(课程→承重,B2 follow≫vanilla/base、other 平)**:**否**。operand-follow B2=vanilla=base=0。
- **H_fail_bypass(read/reconstruct 捷径挺过课程,follow≈0、snap 高、B2≈vanilla)**:**✅ 命中(预判「最可能」)**。
  细化:此 regime 图读不动,故 bypass 具体形态 = **抄链尾结论**(非图表的「重读图」、非自然计数的「从枚举重建」)。
- **H_fail_fabricate(other↑)**:**否**。operand/consistent 的 other 均 ≤.05(base .048)。与 N3「剥夺→重建非幻觉」一致。
- → **「either way a result」兑现**:课程**装不出**承重链 = 对「轨迹蒸馏能内化什么」的**硬边界**,与 masked、N3 三路收敛。

## 5. 与 N3 轴的衔接(组织性发现再 +1 个底物 + 收紧边界)
N3 给的边界:真正承重的内化链需证据 **(a) 不可重新感知 ∧ (b) 不能从链冗余复原**。FinQA 渲染表本是 (a) 的强化版
(base .08,比自然计数 .42 更读不动),且 consistent 已抹掉冗余结论满足 (b)——**按 N3 边界本应见到承重极**。
结果却仍非承重,**因为模型走了第三条 bypass:抄链尾结论**。故 N1 把边界**再收紧**:

> 承重的内化链还需 (c) **答案不能作为「结论行」被直接抄取**——探针必须 corrupt 一个**输入**并要求重算
> (operand-only),才测得到真承重;而一旦这么测,**图表/表格/自然计数/FinQA 四底物 follow 全≈0**。
> 「可重读性→答题信息源」轴的统一结论因此更强:**答题信息源永远是「最省力可读到的那个」(重读图 / 链内冗余 /
> 链尾结论),模型从不重算**——reading-not-reasoning 在内化链里是默认而非例外,连可证承重的 teacher + 去捷径课程都翻不动。

## 6. 范围与局限(方法严谨性,记录在前)
- **预注册先行**:H_success/H_fail_bypass/H_fail_fabricate 于结果前提交;按预注册口径(operand-only = battery_followrate 口径)报判决,consistent 为加严旁证。
- **probe 严格性来自 gold program**:被 corrupt 的操作数经**重执行验证「会翻最终答案」**且非冗余;follow 目标 = 重算后的 injected_final(非操作数自身)。这是 N3 证明必须的严格度。
- **teacher 双臂均已跑**:gold_program(主报数,确定性可证承重)+ **文本推理器(§9,本地 orch 解 402 + answer-conditioned 流畅链)**;两 teacher 类型 operand-follow 同为 0 → 结论非 CoT-格式/teacher artifact。
- **课程 dev n=196→kept 180**:规模适中;若要更强可拉 FinQA train(镜像),设计不变。
- **学生是能重读表的 VLM**(操作数全在表内,filter (e)):故意——使承重结果非平凡、bypass 结果为诚实默认。实测 base 重读极弱(.08),正是 (a) 强化。
- **n_eval 按各模型「答对且有 CoT」子集**(base 21 / vanilla 152 / B2 176);operand-follow=0 在各自子集上均成立,无需配对交集即定论(全 0)。
- **32B 复现 ✅**(prereg「if feasible」已兑现,见 §8):逐格复现 8B(operand-follow 0/175、抄结论签名 170/170),跨尺度成立。
- **红线守住**:不单 claim corrupt 证承重;判决由 operand-only(需重算)+ shuffle 控制 + B2-vs-vanilla 对照 + gold-program 可翻性验证四者共同承载。

## 7. 可插入论文的草稿段(留给 WU-6/N4,勿现在塞进未对账 tex)
> **A shortcut-removing curriculum does not internalize a load-bearing chain (preregistered).** As a
> constructive counter-test we SFT Qwen3-VL-8B on a FinQA curriculum containing *only* problems whose
> answer provably requires a multi-step arithmetic chain (≥2 ops, ≥2 distinct table cells, ≥1
> gold-program-verified corruption-flippable operand, answer not equal to any single cell), with a
> deterministic, provably load-bearing gold-program→NL teacher, against a matched-size vanilla control.
> The curriculum raises accuracy on rendered FinQA tables (.08 base → .58 vanilla → **.67 curriculum**),
> but a gold-program-targeted probe shows the chain remains **not load-bearing**: corrupting a verified
> on-path *input* operand flips the answer in **0/172** curriculum cases (0/147 vanilla, 0/21 base; Wilson
> upper bound ≤.025), the model instead reproducing the chain's stated conclusion. A within-case contrast
> isolates the mechanism — when we additionally rewrite the stated conclusion to the recomputed value, the
> model follows it in 96% of cases (165/172), and **every** such case is one where corrupting the input
> alone left the answer unchanged — i.e. the model **copies the chain's conclusion, never recomputes from
> its operands**. This is a third bypass route (after re-reading the image on charts/tables and
> reconstructing from chain redundancy on natural-image counting), tightening the boundary on what
> trajectory distillation can instill and re-confirming, with a training-time intervention, that accuracy
> and faithfulness are decoupled.

## 8. 32B 跨尺度复现(prereg「if feasible」✅,`battery_n1_32b.json`)
QLoRA-SFT Qwen3-VL-32B(nf4,同 8B 配方:strict 课程 vs none vanilla,各 dev 180,gold_program teacher,
3 ep / lr 1e-4 / r16;**取 epoch_3 与 8B 同口径**,monitor-slice 上 b2/vanilla 准确率对齐 .600)。探针同 `battery_n1_targeted.py`。

| mode | base_acc | n_eval | n_tgt | **operand follow** | operand snap | consistent follow | shuffle follow |
|---|---|---|---|---|---|---|---|
| base    | .123 | 32  | 32  | **0/32 (.000, CI[.000,.107])**  | .844 | .781 | .000 |
| vanilla | .659 | 172 | 168 | **0/168 (.000, CI[.000,.022])** | 1.000 | .976 | .006 |
| **B2**  | .682 | 178 | 175 | **0/175 (.000, CI[.000,.021])** | 1.000 | .971 | .000 |

- **逐格复现 8B**:operand-follow 全 0(8B 0/21·0/147·0/172 → 32B 0/32·0/168·0/175);consistent-follow 高
  (.781/.976/.971);**抄结论同案签名 b2 170/170、vanilla 164/164**(8B 165/165)。→ **课程装不出承重链
  跨尺度成立**,机制(抄链尾结论·不重算)逐格一致。
- **accuracy ⊥ faithfulness 跨尺度**:课程升准确率(.123 base → .659 vanilla → **.682 B2**,B2≥vanilla 且二者≫base
  +53–56pt),operand-follow 仍钉死 0。注:32B 上 B2 与 vanilla 准确率更近(.682 vs .659,二者 epoch_3 在 slice 上同 .600)
  → **准确率对齐**下承重性仍同(both 0),排除准确率混淆。
- **判决不变**:**H_fail_bypass 跨 8B/32B 双尺度命中**;H_success / H_fail_fabricate 双尺度均否(operand other ≤ .156,
  且仅 base 小样本 n=32 偏高,SFT 后 0)。

## 9. 文本-teacher 主臂(prereg ⑤「文本推理器读 GT 表」✅,`battery_n1_8b_text.json`)
预注册把**文本推理器**列为 ⑤ 主 teacher、gold-program 为 robustness 臂;§2–§8 先用 gold-program(因 DeepSeek orch 402)。
本节补上文本臂,**回应「H_fail_bypass 是否只是 gold-program 那种死板『Step k: …』CoT 格式的 artifact」**。

**解 402 + 去 teacher 质量混淆**:按 [[deepseek-402-local-orchestrator]] 把 `ORCHESTRATOR_*` 指向**本地** vLLM
(Qwen3-30B-A3B-Instruct-AWQ @:30001,TP=2)。直接文本 teacher 在 FinQA 多步上只 14% 命中 gold(196→kept 27,
teacher 质量瓶颈,非待测效应)。故改用 **answer-conditioned rationalization**:teacher 收到 gold 答案 + 承重操作数,
写一条**流畅自然语言**链推到该答案——(a) 去 teacher 质量混淆(同 gold-program 臂目的),(b) 链**流畅、非死板格式**
(正是本臂要检验的差异),(c) 构造上承重,(d) **不泄漏给探针**(学生测试期无特权信息)。yield 196→**173**(drift 23)。
样例 CoT:"To find the percentage … 1. Identify the total mmboe for Canada: 60.0 mmboe. 2. Identify the total: 243.0 …"
(**流畅 NL,非 "Step k: multiply"**)。课程操作化同 §验收:strict-text 173(n_ops {2:144,3:16,4:3,5:10},全 ≥2 步)
vs vanilla-text 173(n_ops {1:**105**,2:61,3:6,5:1},含 105 单步捷径),同 teacher、同 n、独立图目录。SFT 同配方(3 ep)。

| mode | base_acc | n_eval | n_tgt | **operand follow** | operand snap | consistent follow | shuffle other |
|---|---|---|---|---|---|---|---|
| base        | .080 | 21  | 21  | **0/21 (.000, CI[.000,.155])**  | .952 | .857 | .095 |
| vanilla-text| .556 | 145 | 142 | **0/142 (.000, CI[.000,.026])** | .993 | .915 | .063 |
| **B2-text** | .640 | 167 | 162 | **0/162 (.000, CI[.000,.023])** | .994 | .963 | .006 |

- **逐结论复现 gold-program 臂**:operand-follow 全 0(0/21·0/142·0/162);consistent-follow 高(.857/.915/.963);
  **抄结论同案签名 b2text 155/156、vanilla-text 130/130**。→ **H_fail_bypass 不是 gold-program CoT 格式的 artifact**:
  连**流畅、可证承重的文本-teacher 链**也装不出承重 student,机制仍是**抄链尾结论·不从操作数重算**。
- **accuracy ⊥ faithfulness 再确认**:课程升准确率(.080→.556→**.640**,B2-text 比 vanilla-text 高 **+8.4 pt**,
  与 gold-program 臂 +9.2 同向),operand-follow 仍 0。
- **汇总:H_fail_bypass 跨 2 尺度(8B/32B)× 2 teacher 类型(gold-program 格式 / 流畅文本推理器)四点共证**——
  课程装不出承重链是**稳健**结论,非格式/尺度/teacher 的 artifact。prereg ⑤ 双 teacher 臂闭环。
