# N3 完成快照 — 自然图(V*-式)极探针:预注册的「干净承重第二极」被否

> 完成 2026-06-25(8B 全条件;32B 跑完后追加,见 §6)。预注册见
> [n3_prereg_0625.md](n3_prereg_0625.md)(**先于结果提交**,019cc76)。
> 所有数字经 `data/distill/poc/battery_natcount8b_{present,masked}.json` 指纹化产出;
> 一键重生:测试集 `python scripts/build_natcount_test.py`;探针网格
> `bash scripts/run_n3.sh`;统一对照表 `python scripts/n3_report.py`
> (→ `data/distill/results/n3_regime.{json,md}`)。
>
> **一句话**:在感知瓶颈的自然图复杂计数上(base free-form acc **.415**,远低于图表/表格的
> .80–.97),预注册的两个假设**都被否**:既没有「干净承重第二极」(`follow`↑;**H_pole 否**),
> 也没有「在场即塌缩」(`other`↑;**H_collapse 否**)。**即便把 CoT 里的计数本身改错、并遮掉图像,
> 模型也不采纳注入的错值(follow 0/43),而是从链里冗余的逐项枚举里把真值找回来。** 于是
> 「reading not reasoning」式的「CoT 数字内容非承重」(F=corrupt−shuffle=0、follow≈0)**复现到第三种底物
> (自然图)**;而 2510.23482 所镜像的「自然图上文本 CoT 承重」在本 regime **不出现**——这是一个
> **预注册接住的、诚实的负结果**(prereg 已写明「干净第二极风险高、否亦是结论」),并把组织性发现从
> 「双/三极」**修正**为更精确的「答题时信息源」轴(见 §4)。

## 验收(对应 todo §N3 + 范围闸)
- [x] 建小自然图子集 → `{case_id,question,gold}` + 图目录(复用探针 harness)
      → `scripts/build_natcount_test.py`:TallyQA **complex** 计数 n=400(自然 VG 图、整数自由作答、
      **无 MCQ→无 letter-luck**),grader 自检 400/400、mcq=0。
- [x] free-form + battery + `battery_followrate.py`(corrupt/shuffle/重读,present+masked)
      → 用**通用基座 8B**(`--adapter none`,非 chart-SFT 学生)跑全套;free-form acc 由 battery
      Phase-1 给出(.415)。
- [x] **先写预注册**(H_pole / H_collapse 两向)→ 已提交于结果之前。
- [x] 验收:写出 verdict + regime 轴 → **两向假设都否**;regime 轴修正为「答题信息源」(§4)。
- [x] (范围闸)V* 保持小,只佐证;不做完整自然图研究 → n=400、单基座、无训练,纯探针。
- [~] 32B 复现(prereg「if feasible」):跑完追加(§6)。

## 1. 底物为何这样选(诚实记录)
- **真 V*Bench 是多选题**(属性/空间关系),与数值 corrupt 探针 + snap/follow/other 三分**不兼容**。
- TallyQA-**complex**(`is_simple=False`)= 自然图上「感知 + 过滤 + 计数」("how many people are wearing
  blue shirts"),是图表「感知 + 算术」的**自然图类比**,且**自由整数作答**→ grader/探针**逐字同口径**。
- n=400,答案 ∈ [2,20](分布 {2:228,3:77,4:46,5:20,…}),388 唯一图,全 AMT 人写复杂题。
- **已记录的局限**:VG 图中位分辨率 ~500px,**不是 4K-0.1%-小目标的极端 V***;故这是「自然图推理-计数」
  regime,而非最极端的分辨率瓶颈。范围闸:佐证用,非完整研究。

## 2. 头条数字(base 8B,n_eval=166,present + masked)
| 条件 | base acc | corrupt | shuffle | **F=c−s** | snap | follow | other | filler |
|---|---|---|---|---|---|---|---|---|
| **present** | **.415** | .030 | .030 | **0.000** | **.988** | **.000** | .012 | .373 |
| **masked**  | .415 | .036 | .036 | −0.000 | .982 | **.000** | .018 | 1.000(acc=0) |

- **F=0、follow=0**:改 CoT 里一个数字翻转不超过打乱整条链,且模型**从不**采纳注入的错值 → CoT 的
  **数字内容非承重**——与图表/表格同向,**复现到第三种底物**。
- truncate 单调(.373/.247/.096)、delete 单调(.036/.042/.066):格式/长度才咬,具体数值不咬(同图表/表格)。
- **present↔masked 几乎不动**(Δflip +.006、Δsnap −.006、Δother +.006)——与图表/表格**形成尖锐对比**(见 §3)。

### 2.1 杀手级诊断:把「计数本身」改错、并遮图,模型仍答真值(回应「corrupt 没打中答案」)
对 corrupt **恰好命中 gold 值那枚 token** 的子集(injected=2·gold+7,如 gold=3→注入"13"):
| 条件 | 命中-gold-token 子集 n | snap→真值 | **follow→注入** | other |
|---|---|---|---|---|
| present | 43 (26%) | **42** | **0** | 1 |
| **masked** | 43 | **42** | **0** | 1 |
- **即使把链里的计数直接写成 13、并把图遮掉,模型仍答 3(42/43),0/43 跟随注入** → `follow≈0` **不是**
  「corrupt 没打中答案」的假象。模型从链里**冗余的逐项枚举**(person 1/2/3…)把真计数找回来 → 注入的单值
  打不过冗余上下文 → **该 CoT 是收据(answer 冗余可复原),不是承重配方**。
- 复现命令:见本节脚注口径(`battery_natcount8b_{present,masked}.json` 的 details:gold/injected/corrupt_ans)。

## 3. 统一 regime 对照(`scripts/n3_report.py`,present 行带 95% Wilson CI)
| cell | cond | n | baseAcc | flip | snap | follow | other | regime |
|---|---|---|---|---|---|---|---|---|
| chart 8b | present | 321 | — | .212 | .816 | .031 | **.153** | fabricate* |
| chart 8b | masked | 321 | — | .333 | .710 | .100 | **.190** | — |
| table 8b | present | 385 | — | .034 | .971 | .008 | .021 | bypass |
| table 8b | masked | 385 | — | .278 | .722 | .088 | **.190** | — |
| **nat 8b** | **present** | 166 | **.415** | .030 | .988 | **.000** | .012 | **bypass** |
| **nat 8b** | **masked** | 166 | .415 | .036 | .982 | **.000** | .018 | — |
| chart 32b | present | 316 | — | .051 | .981 | .016 | .003 | bypass |
| table 32b | present | 389 | — | .033 | .969 | .015 | .015 | bypass |
| **nat 32b** | **present** | 179 | **.448** | .022 | .978 | **.000** | .022 | **bypass** |
| **nat 32b** | **masked** | — | — | — | — | — | — | (跑完追加, §6) |

\* chart-8B-present 的 `other`=.15 偏高(8B 在 ChartQA 上即便有图也偶发幻觉);32B/table 在场都是干净 bypass。

**关键对比 = present→masked 的 Δ(谁是答题时的信息源)**:
- **图表/表格**:遮图 → 行为大变(table 8B flip .034→.278、other .02→.19;chart 8B flip .21→.33)→
  **答题靠重读图像**,遮掉就被迫**幻觉**(fabricate)。
- **自然计数**:遮图 → 几乎不动(flip .030→.036、other .012→.018)→ **答案已冗余编码在 CoT 枚举里**,
  图在答题时基本没被用 → 遮图不逼出幻觉。
- 共同点:**两边的 corrupt 都 ≤ shuffle、follow≈0** → 任一底物上**单个 CoT 数字都非承重**。

## 4. 组织性发现(修正版,诚实)——预注册接住
预注册原计划是「双极/三态」(图表 bypass / 自然图 load-bearing / masked fabricate)。**数据否掉了**
「自然图 load-bearing」这一极:

> **可重读性轴真正控制的不是「CoT 是否承重」(它在三种底物上都不承重),而是「答题时的信息源」**:
> - 图表/表格:信息源 = **重读图像**(遮图 → 被迫幻觉);
> - 自然计数:信息源 = **CoT 自身的冗余记录**(遮图 → 基本不变);
> - 但**两类底物上,被外化链里的单个数字都不是承重的**(F≤0、follow≈0)。
>
> 因此 **2510.23482 的「自然图上文本 CoT 承重」镜像在本 regime 不复现**;一个真正的承重 regime(若存在)
> 需同时满足 (a) 目标**不可重新感知** 且 (b) 答案**不能从链里冗余复原**——中位-500px 的计数两者都不满足。
> 这把「reading not reasoning」讲得更精确,也给「蒸馏/基座能内化什么」划了更硬的边界。

## 5. 范围 / 诚实记录
- **预注册先行**:H_pole/H_collapse 两向于结果前提交(019cc76);两向**都否**,如实报。
- **probe 在计数上偏保守**:计数 CoT 的答案=最终计数,且被逐项枚举**冗余编码**;改单个数字(即便命中计数)
  能被枚举复原 → corrupt 在计数上是**敏感度下界**(同 chapter_plan 红线「不单 claim corrupt」)。但
  **§2.1 的 masked+命中计数仍 0/43 follow** 已把「H_pole(承重)」直接钉死,不靠 corrupt 的灵敏度。
- **base 探针非 chart-SFT 学生**(prereg 依据):chart-SFT 学生在自然图出域→几乎全错→ kept 太少;且
  可重读性问题问的是基座感知策略本身。
- **未做训练**:N3 是纯探针(无 SFT),故无 train/test 泄漏问题,无需 disjoint 检查。
- **底物分辨率局限**(§1):非极端小目标 V*;结论限于「自然图推理-计数」regime。
- paraphrase 干预**跳过**(需 DeepSeek API,对 follow/other 结论非承重);本地控制(filler/truncate/delete)齐全。
- **paper 集成**:paper.tex 仍为升级前数字(n=47/60),全篇数字对账属 WU-6/N4 写作 pass;本快照给出
  可直接插入的 §5.2 草稿(见 §7),**不**单独把 N3 数字塞进尚未对账的 tex。

## 6. 32B 复现(prereg「if feasible」)
- **present ✅ 逐格复现 8B**(n_eval=179,base acc .448):**corrupt .022 = shuffle .022 → F=0**、
  **follow 0/179**、snap .978、other .022;**count-token 诊断 follow 0/60**(直接改错计数也 0 跟随)。
  → 两个尺度都否 H_pole / H_collapse,头条结论跨尺度成立。
- **masked**:32B 链在跑(2×3080 上 32B-nf4 phase-1 较慢),跑完追加本节 + §3 表 `nat 32b masked` 行 +
  `n3_regime.json`。命令现成:`python scripts/n3_report.py` 会自动带出。预期:同 8B 近乎不动(other 平)。

## 7. 可插入论文的 §5.2 草稿(留给 WU-6,勿现在塞进未对账 tex)
> **Natural-image pole (preregistered).** To test whether the masked-chart fabrication is an artifact
> of artificial masking or reproduces on naturally hard-to-re-perceive images, we preregistered a probe
> (H_pole: load-bearing CoT, `follow`↑; H_collapse: fabrication, `other`↑) on TallyQA-complex counting
> (natural images, free-form integer answers, base Qwen3-VL-8B, no chart adapter). Despite a genuinely
> perception-bottlenecked cell (free-form acc .415 vs .80–.97 on charts/tables), **both hypotheses fail**:
> corrupting a CoT number flips no more than shuffling (F=0) and the model never adopts the injected value
> (`follow` 0/166) — even when the count itself is falsified and the image is masked (0/43). The
> internalized chain's numeric content is not load-bearing on a third, natural-image substrate; the
> 2510.23482 "text-CoT-load-bearing on natural images" mirror does **not** reproduce at this resolution.
> The rereadability axis controls the answer-time *information source* (re-read image for charts/tables —
> masking forces fabrication; redundant CoT for counting — masking is inert), but in neither case is a
> single internalized CoT number load-bearing.

## 一句话(给 reviewer 的防线)
自然图(感知瓶颈、base .415)上**预注册**探针:**承重第二极不存在**(follow 0/166,连「改错计数+遮图」也 0/43)、
**在场塌缩也不存在**(other .012);**「外化 CoT 数字非承重」复现到第三种底物**,而「自然图文本 CoT 承重」镜像
在本 regime 不出现——一个被预注册接住的诚实负结果,把组织性发现从「双极」修正为更精确的「答题信息源」轴。
