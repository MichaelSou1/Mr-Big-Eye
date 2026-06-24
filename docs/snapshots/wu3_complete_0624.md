# WU-3 完成快照 — 加 TabMWP(第二个 reasoning-bound 数据集,杀「单数据集」)

> 完成 2026-06-24。所有数字经 `data/distill/results/{results.jsonl,map.json}` +
> `data/distill/poc/{eval_sft_*_tabmwp_n400.json, battery_tabmwp*_{present,masked}.json}` 指纹化产出。
> 一键重生:测试集 `python scripts/build_tabmwp_test.py`;教师 CoT `python scripts/poc_gen_cot_tabmwp.py`;
> gate `scripts/run_chartqa_gate.py --dataset tabmwp …`;SFT+eval+battery `bash scripts/run_wu3_stage2.sh <8b|32b>`;
> MAP `python scripts/regen_tables.py && python scripts/build_map.py`。
> 一句话:**蒸馏 SFT 在 TabMWP 上把准确率真推高(8B +9.8% p<1e-8、32B +13.8% p≈2e-12),且因果探针证明
> SFT 模型的正确答案不依赖 CoT 里的算术(corrupt flip .034 ≪ shuffle .145、改数后几乎不跟随注入错值)——
> 结合两者,内化的是「读」非「承重算」,"reading, not reasoning" 跨 ChartQA 复现。**
> ✅ **「增益=读」已逐例直证(8B+32B)**:增益子集(8B n=40 / 32B n=57)单看 CoT-corrupt → follow_injected 1/40、**0/57** → 新对答案非 CoT 算术所得(§4.1,小样本 caveat)。
> ✅ **算术在哪儿:image-mask 给答案**(§4.2):有图→出答案那步重读重算、无视坏链(follow≈0);**遮图→塌缩成幻觉**(8B "other" .02→.19,9×)而非回退用链 → **承重的是图(感知)不是链**;正面回应"算术该用思维链"。

## 验收(对应 todo §WU-3 + 范围闸)
- [x] `scripts/build_tabmwp_test.py`(镜像 `build_chartqa_test.py`):TabMWP test → `{case_id,question,gold}` + 图目录
- [x] gate(free-form + self_reflect,多 seed)→ 把 TabMWP cell 放上 MAP(**范围闸:orch 延后,见下「范围/诚实」**)
- [x] 在 regime-2 base 上跑 §11 SFT + WU-2 battery(**8B 全套完成**;32B SFT + n=400 eval 完成 **+13.8% p≈2e-12**、battery 待调度)
- [x] **验收**:TabMWP 上 MAP(regime/cell + net±CI)
- [x] **验收**:≥1 个 TabMWP regime-2 cell 的 SFT Δacc + battery → reading-not-reasoning 跨 ChartQA **复现(8B,坐实)**
- [x] (范围闸)时间紧只做 regime-2 cell + probe,不做全 ladder

## 1. 测试集(`scripts/build_tabmwp_test.py`)
- 源:`zyhang1998/tabmwp`(canonical TabMWP,**渲染表格图**内嵌 PNG)`problems_test.parquet`。
- 取 **free-response 数值子集**(`Choices==[]` 且 Answer 数值)——与 ChartQA 同质的开放式数值底物,**无字母选项→无 disguised-accuracy / letter-luck**(正面回应 2402.14897)。
- `data/distill/tabmwp/test_cases_400.jsonl` = **400** 行 `{case_id:"tabmwp-<id>", question, gold}`;图 `tabmwp_<id>.png` × 400。
- **与全部 23,059 张 train 表格图 hash-disjoint(0 leak)**;grader 自检 gold-as-answer **400/400**、mcq=0、letter-luck=0。
- 选型理由:表格**可重读**(值在格里)但答案需**多步算术**(差/和/均值/茎叶计数)→ 正好检验"reading not reasoning"在第二个图表/表格 regime 是否复现;自带 GT 表(与降级的 WU-4 共用底料)。

## 2. Gate + MAP(regime/cell + net±CI)
脚本泛化:`run_chartqa_gate.py / eval_sft_n400.py / battery_n400.py / poc_sft_32b_qlora.py` 改为**从 case_id 推图前缀**
(`chartqa-N→chartqa_N.png`、`tabmwp-N→tabmwp_N.png`),ChartQA 行为逐字不变;gate 加 `--dataset` 标签。

| cell | free_acc | self_reflect net (k=3) | 95% CI | verdict | regime |
|---|---|---|---|---|---|
| tabmwp **8b** | **.927** | **−.014** | [−.033, −.003] | effect(**反而有害**) | **R2** |
| tabmwp **32b** | **.980** | +.003 | [.000, +.010] | within-variance | **R2** |

- **MAP**(`data/distill/results/map.json`):两 cell 都判 **R2(reasoning-bound)**;regime-2 集合现 = `{chartqa/32b, tabmwp/32b, tabmwp/8b}` → **跨两数据集**,杀"单数据集"。
- 规则:free_acc≥.75(表格易读,小模型也过 perception 闸)+ 残差=多步算术 → reasoning-bound。
- self_reflect 在 8B 上**显著有害**(−1.4%,模型二次猜翻对的)、32B 近天花板无效——agentic 在 TabMWP 同样不帮忙(同向 ChartQA)。

## 3. SFT Δacc(`run_wu3_stage2.sh`;teacher=32B 读表算术链 160 条,一致性过滤 98.8%)
| cell | base | SFT | net | 95% CI | McNemar | n_eval |
|---|---|---|---|---|---|---|
| **8b** | .865 | **.963** | **+.098** | [+.070, +.128] | b=40 c=1 → **p=2.9e-9** | 400 |
| **32b** | .835 | **.973** | **+.138** | [+.102, +.172] | b=57 c=2 → **p=2.1e-12** | 400 |

- **关键**:gate free-form .93/.98 像近天花板,但 eval harness 用更严格的 "solve step by step" 基线提示,base 实际 .865/.835 → **SFT 有大空间且效果强、显著**(8B +9.8% p<1e-8、32B **+13.8% p≈2e-12**,均比 ChartQA 同尺度 +6.5%/+7.0% 更大)。
- 32B per-epoch 选 **epoch_4**(60-held 打到 1.000);n=400 正式:base .835→.973、**net +.138 CI[+.102,+.172]、McNemar b=57 c=2 p=2.1e-12**(gain 57 lost 2)。

## 4. Battery(WU-2 全套干预)→ reading-not-reasoning 复现(8B n_eval=385 / 32B n_eval=389)
| 条件 | corrupt | shuffle | **F=c−s** | snap_rate | 解读 |
|---|---|---|---|---|---|
| **present** | **.034** | .145 | **−.112** | **.971**(374/385) | 改 CoT 数字几乎不翻、打乱反而更翻;改数后答案 **97% 吸附回真表值**、跟随注入错值 0.8% → **CoT 非承重,模型重读表格** |
| **masked** | .278 | .299 | **−.021** | — | 遮图后两个 flip 都飙升(图/表是主信息源),但 corrupt 仍**不超** shuffle → CoT 里的数字始终不承重 |

控制对(present):paraphrase **.047 ≈ corrupt .034**(改数 ≈ 改写 → 数字不承重);filler .655 / truncate@.25 .597(格式/长度才咬,单调);delete-k 单调(.060/.219/.331)。

### 跨数据集对比(8B,present / masked)— 逐格同向
| | corrupt | shuffle | F | snap |
|---|---|---|---|---|
| ChartQA present | .212 | .302 | −.090 | .816 |
| **TabMWP present** | **.034** | **.145** | **−.112** | **.971** |
| ChartQA masked | .333 | .408 | −.075 | — |
| **TabMWP masked** | .278 | .299 | **−.021** | — |
→ **TabMWP 上 8B 比 ChartQA 还更"读不在算"**:改算术数字只翻 3.4%,97% 吸附真值。

### 4.0 32B battery(四卡并行 present∥masked,n_eval=389)— 跨尺度同向
| 条件 | corrupt | shuffle | **F** | snap | follow |
|---|---|---|---|---|---|
| 32B present | .033 | .072 | **−.039** | .969 | .015 |
| 32B masked | .075 | .103 | −.028 | — | — |
→ 32B 逐格复现 8B/ChartQA:present corrupt≪shuffle、snap .97、follow .015 → CoT 非承重,两尺度两数据集一致。

### 4.2 算术在哪儿发生?image-mask 的 follow/other 分解(回应"算术该用链"的诘问)
对 present vs masked 逐例分类 corrupt 后的答案(`/tmp/masked_follow` 口径,由 battery details 复算):
| cell/cond | flip | snap→真值 | **follow→注入** | **other(乱编)** |
|---|---|---|---|---|
| 8B present | .034 | .971 | .008 | .021 |
| **8B masked** | .278 | .722 | **.088**(11×) | **.190**(9×) |
| 32B present | .033 | .969 | .015 | .015 |
| **32B masked** | .075 | .928 | **.039**(2.6×) | .033 |

- **机制定位**:**有图** → 出答案那步**回去重读两格、当场重算**,把写坏的链当空气(follow≈0、snap .97)。**遮图** → 没图可读,**真正暴涨的是 "other"(乱编,8B .021→.190 9×)** 而非 follow(.088)—— 即模型**不是乖乖回退用 CoT 的数,而是对图里数据产生幻觉**(用户假说,实测坐实)。
- **正面回答"算术该用思维链"**:算术真实发生,但**不由外化链承载**——有图时在答案步从感知重算,遮图时塌缩成幻觉。**承重的是图(感知),不是链;链是收据不是配方。** 32B 更强、幻觉少(masked other 仅 .033、snap 仍 .928),8B 弱则崩得狠 → 同一机制的强弱版。
- 这把"reading not reasoning"讲精确:迁移的是**感知(重读)**,算术摊进前向、不挂在 token 上;**红线守住**——不当主句 claim「内化推理不承重」(2602.22766 占位),只 claim *外化链*非承重 + 迁移的是读。

### 4.1 「增益=读」逐例直证(`scripts/decompose_gain.py`,纯 CPU,复用已有产出)
把 battery 的 SFT-正确 kept 集**按 *pre-SFT base* 是否答对**切两半,看 CoT-corrupt 行为(join `eval_*.preds.jsonl` 的 base_correct + battery 的 per-case corrupt_ans):

| cell | 分区 | n | corrupt_flip | **follow_injected** | snap_to_gold |
|---|---|---|---|---|---|
| **8B** | **GAINED**(base 错→SFT 对,*即 +Δacc 子集*) | **40** | **.000**(0/40) | **.025**(1/40,95%≤.129) | **1.000**(40/40) |
| 8B | RETAINED(base 对→SFT 对) | 345 | .038 | .035 | .968 |
| **32B** | **GAINED** | **57** | **.018**(1/57) | **.000**(0/57,95%≤.063) | **.982**(56/57) |
| 32B | RETAINED | 332 | .036 | .042 | .967 |

- **直证(两尺度)**:新增对题里,改掉 SFT 自己 CoT 中一个数字 → **8B 0/40 翻、32B 1/57 翻;follow_injected 8B 1/40、32B 0/57** → 新对答案**不是从 CoT 算术算出来的**,是读表得来的。**「增益来自读」从推断升级为逐例陈述**,32B 增益子集(n=57,follow 95% 上界 .063)比 8B 更干净。
- **caveat**:小样本(n=40/57),follow Wilson 95% 上界 .129/.063;corrupt 随机选数 → flip 是敏感度下界(chapter_plan 红线「不单 claim corrupt」)。
- 产出:`data/distill/poc/gain_decomp_{8b,32b}_tabmwp.json`。

## 范围 / 诚实记录
- **「增益来自读」已逐例直证(8B+32B,见 §4.1)**:增益子集(8B n=40 / 32B n=57)单看 CoT-corrupt——follow_injected 1/40、0/57 → 新对答案非 CoT 算术所得。比 WU-5 的*聚合* faithfulness-axis 更直接(逐例)。原 WU-5 ChartQA 聚合结论(F≈0 不随 SFT 变)与此一致。
- **snap_rate 与 corrupt_flip 非独立**:snap(.971)≈ 1−corrupt_flip(.034),真正独立的信号是 **follow_rate=.008**(改数后答案几乎不跟随注入的错值)+ **corrupt≪shuffle**。报 snap 是为与 ChartQA(WU-2)同口径对比,别把它当第二个独立证据。
- **corrupt 随机选 CoT 里一个数**:若选中非承重的中间值,即便忠实推理器也不翻 → corrupt-flip 是敏感度**下界**(chapter_plan 红线「不单 claim corrupt」)。shuffle 对照 + 跨数据集一致性缓解,但不消除。
- **orch_reflect_blind 延后**:32B-AWQ 在 2×3080 上 ~1.1–1.7 req/s,orch(DeepSeek 推理模型,每例多调)× n=400 × seeds 需数小时;MAP 验收只需 free_acc + 一个 agentic net±CI(self_reflect 已给)。按**范围闸**用 self_reflect 表征 within-variance,orch 留作可选补充(脚本现成:`--methods orch_reflect_blind`)。
- **32B battery 已完成**(四卡并行 present∥masked,2026-06-24 22:21):present F=−.039/snap .969/follow .015、masked F=−.028,逐格复现 8B(§4.0);present∥masked 用分离 paraphrase 缓存避免并发写竞争(base-CoT 都带图,内容一致)。
- **masked follow/snap 为 present-only 自动产出 + masked 由 details 复算**(`/tmp/masked_follow` 口径);follow/other 在 masked 上升是机制证据(§4.2),非 bug。
- **paraphrase 数字保真**:8B 231/385(60%)、32B 229/389(59%,DeepSeek 长链漂数);漂移只**抬高** paraphrase flip,而实测 paraphrase ≤ corrupt → 报出值为**保守上界**,控制成立(同 WU-2)。
- gate seeds=3(ChartQA 用 5);第二数据集按范围闸取 3,pooled CI 仍 case+seed 双重重采样。

## 一句话
TabMWP(表格可重读、需多步算术)上:**SFT 真把准确率推高(8B +9.8% p<1e-8、32B +13.8% p≈2e-12)→ battery 证明 CoT 非承重(F=−.112、corrupt≈paraphrase),且逐例分解坐实那 40 个增益题里 0 翻/40 snap/1 follow → 增益本身=读表非算 → "reading, not reasoning" 跨 ChartQA 复现、TabMWP 更极端**。regime-2/reasoning-bound 现跨两数据集三 cell,reviewer 的"单数据集"攻击被堵死。
