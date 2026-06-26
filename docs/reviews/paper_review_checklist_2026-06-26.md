# Reading, Not Reasoning — 修订 Spec + 执行清单

日期：2026-06-26（数据落地版）
对应主稿：`docs/paper/paper.tex`
结果库：`data/distill/poc/*.json`（append-only），脚本见 `scripts/`

本文件是可执行修订规范：每条含 **目标 / 位置 / 现状证据 / 动作 / 验收**，按优先级排序。
所有 `〔✓盘上〕` 标注的数据已用结果库核验，编辑时直接引用，勿凭草稿数字。

---

## 0. 关键决策（已定）

- **标题**：保留 `Reading, Not Reasoning`，在 intro + discussion 各加一句 caveat，把因果结论限定到 *emitted written chain*，承认 internal computation 不可证伪。**不**因 1A 改标题。
- **LoRA 替代解释**：用 full-FT 对照**堵死**，**不**把结论收紧到 "under LoRA/QLoRA"。
- **preregistered**：若无真 OSF/AsPredicted 锚点，降级为 `pre-specified`；**禁止**事后硬凑锚点（诚信红线）。

## 0.1 盘上已验证的三个事实（驱动下面的修订）

1. **8B dense/full-SFT 对照已大体落盘，TabMWP full battery 还差 GPU run**：训练为 dense control（视觉塔不训，embedding + 前 3 层冻结；ChartQA/TabMWP 可训约 79.7% 参数），不是 literally all-parameter FT。ChartQA eval acc `.7125→.7625`(+5.0, McNemar p=.0126) 且六 intervention battery 已完成；TabMWP eval acc `.865→.9675`(+10.3, McNemar p=1.06e-9)，core battery 已有 `snap .971 / follow .016 / corrupt-flip .049`；FinQA 四个 targeted Full-SFT arms 已完成。TabMWP present/masked 六 intervention battery 需等 GPU 空出后补跑，不能先声称全部 complete。
2. **item 5 草稿数字错**：8B gain-subset 是 `follow 1/40`（上界 12.9%），不是 `0/40`(8.8%)；32B `0/57`（上界 6.3%）。CI 已在 `gain_decomp_*_tabmwp.json` 的 `follow_ci95_hi`。
3. **item 8 不是纯写作编辑**：snap/follow 只在 CORRUPT 下记录；shuffle 不注入值 → "follow under shuffle" 无定义。需从 per-case `details` 重算 snap/other-under-shuffle，或用 shuffle `acc_after` 代理。

---

## P0 — 不修会被拒

### P0-1 标题/“reasoning”过度声明加 caveat 〔新 1A〕

- 位置：title、`paper.tex` intro（A1/A2 后）、discussion。
- 现状：probe 只测 emitted token 因果；`F≈0 + snap≈.97` 严格只支持“written chain 非 causal substrate”，无法排除 activations 内部计算（TabMWP 是真算术，答案必在某处算出）。自引 Pfau / latent-imagination 恰证计算可发生在不可读处。
- 动作：
  - [x] 保留标题不动。
  - [x] intro 加一句：本文所有因果结论针对 *emitted written chain*；不声称无内部计算。
  - [x] discussion 呼应一句：internal computation 不可证伪是方法边界。
  - [x] 全文 framing 不要把 “written-chain unfaithfulness” 写成 “reasoning absence”。
- 验收：reviewer 无法用“你混淆 written-chain 与 reasoning”一句击穿。

### P0-2 full-FT 对照堵死 LoRA 替代解释 〔合 4A〕 〔部分✓盘上；TabMWP full battery 待 GPU〕

- 位置：`paper.tex` Result II / Result III，新增一行/一段 full-FT 对照。
- 现状：现全部 SFT 为 LoRA/QLoRA；reviewer 可主张低秩容量不足才装不进 load-bearing chain。
- 盘上证据（8B dense/full-SFT，`quant:"none"`；视觉塔不训，embedding + 前 3 层冻结）：
  - TabMWP acc：`eval_full_sft_8b_tabmwp_n400.json` → base `.865` → `.9675`，net +10.3，McNemar p=1.06e-9，gain 42 / lost 1。
  - ChartQA acc：`full_8b_chartqa/eval_n400.json` → base `.7125` → `.7625`，net +5.0，McNemar p=.0126，gain 39 / lost 19。
  - TabMWP core faithfulness present：`battery_full8b_tabmwp_present_core.json` → base-free-form acc `.9625`，corrupt-flip `.049`，snap `.971`，follow `.016`。
  - TabMWP core faithfulness masked：`battery_full8b_tabmwp_masked_core.json` → corrupt-flip `.255`，acc_after `.771`（masking 抬 flip，与主结论一致）。
  - ChartQA full battery：`battery_full8b_chartqa_present.json` / `..._masked.json` → six interventions complete；present corrupt-flip `.100`，snap `.917`，follow `.030`；paraphrase flip `.049`。
  - FinQA targeted Full-SFT：`battery_n1_full8b_finqa_{b2,vanilla,b2_text,vanilla_text}.json` → operand-follow ≈0，consistent-follow `.947–.982`。
  - 当前审计与表格：`scripts/audit_full_sft_8b_nonvideo.py` 已生成 `docs/reviews/full_sft_8b_nonvideo_audit.md` 与 `data/distill/poc/full_sft_8b_nonvideo_audit.json`；`scripts/export_full_sft_8b_nonvideo_evidence.py` 已生成 `docs/reviews/full_sft_8b_nonvideo_control_table.md`、`docs/reviews/full_sft_8b_tabmwp_resume_manifest.md`、`data/distill/poc/full_sft_8b_tabmwp_resume_manifest.json`。审计已逐 arm 对齐 6 个原 8B LoRA non-video source arms（ChartQA、TabMWP、FinQA b2/vanilla、FinQA text b2/vanilla）；`source_lora_8b_arms_found` PASS，`lora_to_full_sft_8b_coverage` 仍 MISSING，唯一实际缺口为 TabMWP Full-SFT probe。resume readiness 已写入审计：Mimo paraphrase cache `370/387`，base-CoT cache `0/400`，TabMWP weight shard 保留 `16.33GB`。2026-06-26 16:28 已跑 CPU-only readiness precheck（`PRECHECK_ONLY=1 REQUIRE_GPU_IDLE=0 ...`）并通过：Mimo host/model/key OK，`/home/gpus` free `84GB`，checkpoint 存在；`scripts/finalize_full_sft_8b_nonvideo.sh` 是 GPU run 后的 CPU-only 最终门，当前按预期失败。
  - 结论（当前可写）：dense/full-SFT controls 与 LoRA 同签名，written chain 仍非 load-bearing；但 TabMWP 六 intervention present/masked battery 尚未完成，最终表格等 GPU run 后定稿。
- 动作：
  - [x] 在正文加 dense/full-SFT vs LoRA 对照（至少 8B-TabMWP 一行），明写“dense/full-SFT 不改变 F≈0/high-snap/low-follow”。
  - [ ] **等 GPU 空出**：先跑 `PRECHECK_ONLY=1 bash scripts/resume_full_sft_8b_tabmwp_battery.sh`，确认进程/磁盘/GPU guard 通过；再补跑 `battery_full8b_tabmwp_present.json` 与 `battery_full8b_tabmwp_masked.json` 的六 intervention battery；跑完后逐 cell 拉 `battery_full8b_chartqa_*`、`battery_full8b_tabmwp_*`、`battery_n1_full8b_finqa_*`，扩成多 cell 对照表。
  - [ ] 跑完后执行 `bash scripts/finalize_full_sft_8b_nonvideo.sh`；该命令内部会先产出 TabMWP posthoc shuffle/filler/paraphrase answer classification，再运行 strict audit、重新导出 evidence/manifest，并检查 TabMWP 大权重在 full battery 完成后已清理。strict audit 现在也要求 TabMWP full battery 为新版输出（`details[].answers` 存在），且 posthoc ready 并通过源文件 fingerprint 校验（防 stale posthoc）；该命令 PASS 才能把 P0-2 标成完成。
  - [x] 明确这是 confound-killer，不是把结论降级到 LoRA 范围；同时脚注/表注交代 dense control 的冻结范围。
  - [ ] commit 当前未入库的 full-FT 产物与 eval（`full_8b_*/`, `eval_full_sft_8b_tabmwp_n400.json`, `battery_full8b_*_core.json`）。
- 验收：LoRA-容量替代解释被一行数据驳回；标题可保持强声明。

### P0-3 F 的测量对象与 accuracy 对齐 〔你 P0#3〕

- 位置：`paper.tex:157-160`, `164-184`，Fig 2 caption。
- 现状：Fig 2 caption 写 “F measured on base model”，但 probe 明显跑在 SFT cell 上，accuracy 与 faithfulness 像不在同一对象比较。
- 动作：
  - [x] 确认每个 probe 对应的 checkpoint（base / adapter-loaded student / full-FT student）。
  - [x] 统一到同一 student，或在图注/正文写清 base↔student 对应关系。
- 验收：读者能确定 accuracy 与 F 测于同一模型对象。

### P0-4 Figure 1 图文一致 〔新 5A〕

- 位置：Fig 1 + caption。
- 现状：caption 称 reasoning-bound cells 含 32B ChartQA + 8B/32B TabMWP，但图仅 CHARTQA/NEXT 两列，TabMWP/CLEVRER/FinQA/TallyQA 未画；多 cell 空白。
- 动作：
  - [x] 补 TabMWP 列（及实际覆盖的 dataset），或改 caption 只描述图中实际内容。
  - [x] 处理空白 cell（internvl/8b_sft）的呈现。
- 验收：caption 每一项都能在图中找到对应 cell。

### P0-5 判决性 null 子集 surface CI 〔新 3A〕 〔✓盘上〕

- 位置：`paper.tex` Result III finding 2。
- 现状：草稿裸报 `1/40`、`0/57`，是 under-powered overclaim；且草稿把 8B 写成 0/40（实为 1/40）。
- 盘上证据：`gain_decomp_8b_tabmwp.json` GAINED n=40 follow=1 → `follow_ci95_hi=0.1288`；`gain_decomp_32b_tabmwp.json` GAINED n=57 follow=0 → `follow_ci95_hi=0.0631`。
- 动作：
  - [x] 纠正 8B 数字为 `follow 1/40`。
  - [x] 每个裸 follow/snap 分数后补 95% 上界：8B → 12.9%，32B → 6.3%（直接引 store 里的 `follow_ci95_hi`，勿重算）。
- 验收：“gain subset is reading” 声明带 CI，不再裸报。

### P0-6 Free / base / SFT acc 口径统一 〔合 P0#1〕 〔✓已 trace；待改文〕

- 位置：`paper.tex:38-40`, `102-120`, `145-149`，Table 1 表注，Abstract。
- 现状：Table1 ChartQA-32B `Free acc .798` vs Result II `base .725 → SFT .795` 三数并存。已 trace：
  - `.798` = `data/distill/results/tables.json` / `map.json` 中 `chartqa/32b` free-form 最大 n 口径，来自 `results.jsonl` 的 ChartQA/32B/free_form n=400，`319/400=.7975`，由 `scripts/regen_tables.py` 聚合。
  - `.725 → .795` = `data/distill/poc/lora_32b_chartqa/eval_n400.json` 的 paired SFT eval，base `290/400=.725`，SFT `318/400=.795`，gain 37 / lost 9，McNemar p=6.86e-5。
  - `data/distill/poc/lora_32b_chartqa/base_eval.json` 是旧 n=60 pilot（42/60），不要误引。
- 动作：
  - [x] **先 trace**：定位 ChartQA-32B `.725/.795/.798` 各自来自哪条评估链（base / SFT student / 哪个 split / 哪种 grading）。
  - [x] Table 1 表注写清 `Free acc` = gate/free-form result-store 口径；若表内与 Result II SFT eval 并列，改名为 `Gate free acc` 或拆表。
  - [x] Abstract `.795` 与 Table1 `.798` 不要硬取齐；正文显式说明一个是 SFT eval student acc，一个是 gate/free-form baseline acc。
- 验收：同一 (dataset, model) 的 acc 数字全文自洽，口径可追溯。

---

## P1 — 概念 / 说服力

### P1-7 任务选择循环性写进 Limitations 〔新 1B〕

- 位置：`paper.tex` Result IV / Limitations。
- 现状：ChartQA/TabMWP/TallyQA/FinQA 的 operand 都 input-readable，证“答案靠读”接近 by construction；FinQA curriculum 是最佳回应但 operand 仍在 table。
- 动作：
  - [x] 显式承认：所有 substrate 的 operand 均 input-readable，是 scope boundary。
  - [x] 说明不补“operand 不可读”任务的理由（代价/聚焦），主动定界而非被发现。
- 验收：循环性被作者先认领。

### P1-8 负 F 双刃性单向化 〔合 3B / 你 P1#4〕 〔部分✓盘上；per-case shuffle 待新 battery〕

- 位置：`paper.tex:176-181`, `188-192`，Result III。
- 现状：⚠️ snap/follow 仅在 CORRUPT 下记录；shuffle 不注入值，“follow under shuffle” 无定义。已有 battery summaries 只有 shuffle `acc_after`，旧 JSON details 没保存 shuffle per-case answers；`scripts/battery_n400.py` 已补 `details[].answers`，下次 TabMWP full battery 会自动落 per-case shuffle/filler/paraphrase/truncate/delete answers。
- 反制论证（要赢的方向）：shuffle 破坏的是 parseability 而非语义——证据是 shuffle 后答案仍大量 snap-to-true（full-FT present shuffle `acc_after .857` vs corrupt `.971`），未走向链条诱导的新错值。
- 动作：
  - [ ] 等新 TabMWP full battery JSON 产出后，从 `details[].answers.shuffle` 重算 shuffle 下 snap/other（写入结果库新字段）；旧 ChartQA/TabMWP core 若不重跑，只能在正文用 `acc_after` 作 snap 代理并说明 follow-under-shuffle 不适用。
  - [x] 正文显式报 shuffle 的 snap/acc_after，把负 F 解释为“shuffle 是更强的 *format* 扰动 → F 保守低估”。
  - [x] 避免“follow low under shuffle”这种无定义表述。
- 验收：负 F 只能支持本文结论，无法被反用为“模型对链条结构敏感=有 faithfulness”。

### P1-9 三分法 + Result II→III 桥接 〔你 P1#5/#6〕

- 位置：Abstract、Intro（A1/A2 后）、Result II 段首、Result IV。
- 动作：
  - [x] Abstract/Intro 先讲清三类 answer source：pixels / redundant text / conclusion copying。
  - [x] Result II 段首加桥接句：SFT 增的是更好的 readout，不是更强 agentic reasoning。
  - [x] FinQA 写成“第三种信息源(conclusion copying)”，不是推理失败反例。
- 验收：reading/reasoning/conclusion-copying 三分均衡，FinQA 不显得像例外。

### P1-10 多重比较交代 〔新 3C / 你 #9〕

- 位置：setup + results。
- 动作：
  - [x] 一句话说明 McNemar 只用 discordant pairs，并交代是否做 Holm/BH 校正。
  - [x] 保留 discordant pair 数；说明主 p 值(6.9e-5, 2.1e-12)校正后仍显著。
- 验收：统计严谨性问题被一句话堵掉。

### P1-11 preregistered 降级 〔合 #8〕

- 位置：`paper.tex:59`, `202`。
- 动作：
  - [x] 有真锚点 → 给可核验链接；无 → 改 `pre-specified` / `pre-registered analysis plan (internal)`。
  - [x] 禁止事后硬凑锚点。
- 验收：用词与实际注册状态一致。

---

## P2 — 完整性 / 呈现

### P2-12 核心量提前定义 〔新 2B/2C〕

- [x] corrupt/shuffle/snap/follow + load-bearing 一句话可操作定义，提到 setup 或 Table 2 caption（早于 Table 2 出现处）。
- [x] load-bearing 定义建议：corrupt flip 显著 > shuffle control 且 follow 显著 > 0。

### P2-13 teaser 图 + 浮动体位置 〔新 5B/5C〕

- [x] 加一张 CORRUPT/SHUFFLE/snap/follow 在一条 CoT 上做了什么的 schematic。
- [x] Fig 1 靠近 §4、Fig 2 靠近 §6，勿堆在 References 后。

### P2-14 Table 1 拆斜杠记法 〔新 2A〕

- [x] 表头标注 slash 位 = 4B/8B/32B。
- [x] 异构行（orch −.163、CLEVRER .43–.50）拆出单列/单行，勿与三连斜杠混排。

### P2-15 teacher 单一性 〔合 4B〕

- [x] limitations 再明确一句：主 teacher 为 Qwen3-VL，FinQA text-teacher arm 缓解 format-specific 担忧。

### P2-16 投稿格式 / 匿名 〔你 #10〕

- 备注：当前仓库未发现 `aaai*.sty/cls/bst`，本机 PATH 也无 `pdflatex`/`latexmk`/`tectonic`；AAAI 双栏模板需模板文件到位后再切。
- [ ] 切 AAAI 官方双栏模板。
- [x] 匿名化作者与邮箱。
- [ ] 双栏后检查图宽/表宽。

---

## 执行顺序

1. **先做非 GPU 收尾** → trace P0-6、重算/整理 P1-8、更新 P0-2 表格草稿；不要启动新的 GPU run。当前非 GPU 审计入口为 `/home/gpus/anaconda3/envs/mbe-up/bin/python scripts/audit_full_sft_8b_nonvideo.py --out-json data/distill/poc/full_sft_8b_nonvideo_audit.json --out-md docs/reviews/full_sft_8b_nonvideo_audit.md`；控制表/manifest 入口为 `/home/gpus/anaconda3/envs/mbe-up/bin/python scripts/export_full_sft_8b_nonvideo_evidence.py`；TabMWP battery 后处理入口为 `/home/gpus/anaconda3/envs/mbe-up/bin/python scripts/summarize_full8b_tabmwp_posthoc.py`，单独验收入口为 `/home/gpus/anaconda3/envs/mbe-up/bin/python scripts/summarize_full8b_tabmwp_posthoc.py --strict`。2026-06-26 16:28 已完成 CPU-only precheck、py_compile、bash -n、posthoc pending 检查、strict audit pending 检查与 evidence/manifest 刷新；当前后处理会显示 pending，因为新 full battery JSON 尚未产出 `details[].answers`；新 full battery 产出后会自动计算 shuffle snap/other。
2. **等 GPU 空出** → 先执行 `PRECHECK_ONLY=1 bash scripts/resume_full_sft_8b_tabmwp_battery.sh`。precheck 不加载模型；默认要求相关实验进程为空、`/home/gpus` 至少 40GB 空闲、选中 GPU 单卡 used ≤ 2048MB 且 free ≥ 16000MB，并确认 `.env` 的 `ORCHESTRATOR_API_BASE_URL`/`ORCHESTRATOR_API_KEY`/`ORCHESTRATOR_MODEL_NAME` 已配置且 base URL 包含 `xiaomimimo.com`。通过后执行 `bash scripts/resume_full_sft_8b_tabmwp_battery.sh`，补跑 TabMWP six-intervention present/masked battery（`battery_full8b_tabmwp_present.json`、`battery_full8b_tabmwp_masked.json`）。完成后脚本会清理 `full_8b_tabmwp/model.safetensors`（除非 `KEEP_CHECKPOINTS=1`），并调用 `bash scripts/finalize_full_sft_8b_nonvideo.sh` 做 CPU-only 最终验收和 posthoc answer classification。
3. **只读 trace**（不碰 tex）：P0-6 ChartQA 口径来源已完成；P1-8 shuffle snap/other 等新 battery details。
4. 数据齐后一次性落 P0 文本编辑（P0-1/3/4/5/6）。
5. P1 framing（7/9/10/11）→ P2 呈现（12–16）。
6. 最后做格式/匿名（P2-16）。

## 待办数据依赖（阻塞项）

- [ ] TabMWP Full-SFT six-intervention present/masked battery 跑完 → P0-2 多 cell 表定稿；恢复入口为 `PRECHECK_ONLY=1 bash scripts/resume_full_sft_8b_tabmwp_battery.sh` → `bash scripts/resume_full_sft_8b_tabmwp_battery.sh`。当前 Mimo paraphrase cache `paraphrase_cache_full8b_tabmwp_mimo.jsonl` 为 370/387，base-CoT cache `paraphrase_cache_full8b_tabmwp_mimo_base_cot.jsonl` 为 0/400；缺的 base-CoT 必须由 GPU 上的 Full-SFT 模型生成，之后脚本会用 Mimo API 补齐 paraphrase 并继续 present/masked probes。
- [ ] 最终化 `bash scripts/finalize_full_sft_8b_nonvideo.sh` 通过（尤其 `lora_to_full_sft_8b_coverage` PASS、`tabmwp_full_battery_has_answers` PASS、`tabmwp_posthoc_ready` PASS，且 `tabmwp_weight_clean_after_battery_done` PASS）→ 可声明非视频 8B Full-SFT replicas + matching causal probes 全部完成。
- [x] trace ChartQA-32B `.725/.795/.798` 评估线 → 解锁 P0-6 文本编辑。
- [ ] 重算 shuffle 下 snap/other（写回结果库；需新 battery details 或重跑旧 cell）→ 解锁 P1-8。
