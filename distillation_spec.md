# Mr-Big-Eye Orchestrator 蒸馏 / Agentic SFT 实施方案（项目实测版）

> **目标读者**：Claude Code CLI（可读写本仓库，可在远程 GPU server 上跑训练）
> **本文与 `orchestrator_distillation_spec.md` 的关系**：那份由 claude.ai 离线撰写，
> 未见项目最新代码，存在多处事实偏差（工具数量、guard 机制、数据集、harness 字段、
> 数据量假设等）。**本文以当前代码为准（截至 AGENT_CODE_VERSION = v22，commit 8ba4a6b），
> 取代那份的 Phase A/B 与领域决策部分**；训练侧（Phase C/D 的 LLaMA Factory / vLLM /
> 4×3080 配置）沿用那份已针对 GPU server 调过的数值，仅做对齐说明。
> **终极意图**：以本项目 agent 框架为基座做 agentic SFT，再视情况上 RL。

---

## 0. 现状校准（必读，决定后续一切）

claude.ai 那份 spec 的关键事实偏差，逐条纠正（均已对代码核实）：

| 那份的说法 | 项目实际 | 出处 |
|---|---|---|
| 10 个工具 | **14 个工具**（`multimodal_vqa` 是 legacy，不在注册表） | [tools.py:1151](app/tools.py#L1151) `TOOLS` |
| guards = dedup/stall/salvage/cap | 实为 **dedup / stall / salvage / cap / empty-coercion + 两类 forced call**；`agent_terminated ∈ {None, "cap", "empty"}`（**没有** "stall"/"normal"） | [graph.py:90-169](app/graph.py#L90) |
| 蒸馏在 NExT-GQA 1000 case | 项目最新、最成熟的是 **audiovisual Video-MME 混合集**（199 case）；NExT-GQA 只有 ~20 case 的 fixture 且 pass 率仅 0.35–0.70 | 见 §3 |
| harness 已存 `messages`/`guards_triggered`/`save-full-trajectory` flag | **都没有**。当前每 case 只存 `agent_actions`（工具名序列）等聚合字段 | [eval_harness.py:1406](app/eval_harness.py#L1406) |
| bump `AGENT_CODE_VERSION` 在 .env | 在 [eval_fingerprint.py:13](app/eval_fingerprint.py#L13)；改 orchestrator/QA prompt 会**自动**失效缓存 | 同上 |
| `slice = msgs[:i]` 即训练输入 | **不对**：系统 prompt 每轮现拼、不在 `state.messages` 里；且 forced-call 注入的 AIMessage 不是 teacher 策略 | [graph.py:129-132](app/graph.py#L129)、[graph.py:302-328](app/graph.py#L302) |

**核心 insight（与那份一致，仍成立）**：蒸馏的是 **orchestrator 的工具调用决策**
（输入 = system+history+tool_results，输出 = 下一个 tool_call 或 final answer），
**不是 VLM 的视觉能力**。当前 orchestrator 在 audiovisual smoke 上 `agent_loop_pass_rate = 0.98`，
编排能力已成熟；剩余失败多为 VLM 推理（情绪/时序/选项矛盾），且会被 Tier-3 过滤丢弃，
**不影响蒸馏目标，无需在训练前继续优化**。

---

## 1. 路线图

```
Phase A: 采集 teacher 轨迹  → 扩 harness 落盘完整 trajectory + guards，跑 teacher 存 raw
Phase B: 过滤 + 样本化      → Tier 分级过滤、补系统 prompt、剔除 forced-call target、截断 base64
Phase C: SFT 训练           → LLaMA Factory + LoRA on Qwen2.5-7B（GPU server，配置见 §6）
Phase D: 部署 + 评测        → vLLM serve LoRA，按视频切分的 held-out 集对比 teacher
Phase E:（可选）RL/DPO      → 仅当 Phase D 未达标；注意奖励信号的软门陷阱（§8）
```

**严格按 Phase 顺序，每个 Phase 跑通验收再进下一个。** 但本项目特殊性：先用现有
199-case 混合集走 **pilot 规模**端到端跑通 A→D（趟平管线/bug），**并行**扩 Video-MME
构建更大语料，地基稳了再上正式训练（理由见 §3.3）。

---

## 2. 关键架构事实（蒸馏管线据此设计）

### 2.1 14 个工具（蒸馏的动作空间）

`TOOLS` 见 [tools.py:1151](app/tools.py#L1151)，bind 进 orchestrator（[graph.py:88](app/graph.py#L88)）：

```
retrieve_video_evidence      retrieve_transcript_evidence  search_transcript_keyword
retrieve_slide_evidence      align_audiovisual_evidence    build_timeline
retrieve_hypothesis_evidence segment_focus                 expand_temporal_evidence
stitched_verify              assess_evidence_sufficiency    answer_with_evidence
verify_grounding             search_user_memories
```

（`multimodal_vqa` 定义了 `@tool` 但**不在** `TOOLS`，是旧的一站式兼容工具，忽略。）

工具 schema 是 ground truth，**禁止修改 `tools.py:TOOLS`**——改了等于训练数据作废。
Phase B 用 `model.bind_tools(TOOLS)` 同源导出 schema，保证训练/推理一致。

### 2.2 Agent loop 与 guards（决定 Tier 过滤与样本剔除）

图结构（[graph.py:66-84](app/graph.py#L66)）：`START → orchestrator → (tool_node ⇄ orchestrator)* → memory_write_node → END`。
orchestrator 节点（[graph.py:90](app/graph.py#L90)）每轮做这些**确定性兜底**（这就是"guards"）：

| guard | 触发条件 | 行为 | 可观测信号 |
|---|---|---|---|
| **cap** | tool 调用数 ≥ `orchestrator_max_tool_calls`(=8) | 强制 answer 或 salvage | `agent_terminated="cap"`（无 salvage 时） |
| **empty** | 模型连续返回空 content 且无 tool_calls | 注入 coercion 重试一次 | `agent_terminated="empty"` |
| **stall** | 最近两条 `verify_grounding` 答案相同 | salvage draft 直接结束 | 末尾 AIMessage == 某条 draft |
| **dedup** | 新 tool_call 的 (name,args) 与历史/同批重复 | 合成缓存 ToolMessage、删除该 call | ToolMessage 内容与早前重复 |
| **salvage** | 上述多处 | 取最长未截断的 `answer_with_evidence` 输出 | — |
| **forced answer** | sufficiency=true / grounding 需 revise / cap | 注入 `id="force_answer_with_evidence"` 的 AIMessage | **该 id** |
| **forced verify** | answer 后 `next=verify_grounding` | 注入 `id="force_verify_grounding"` 的 AIMessage | **该 id** |

**两条 forced-call 的 AIMessage 是 harness 注入、不是 teacher 模型生成的**
（[graph.py:302-328](app/graph.py#L302)），样本化时**必须排除为 target**（§5.3），否则学到的是
兜底行为而非编排策略。

### 2.3 GraphState（轨迹与证据都在这里）

[graph.py:36-54](app/graph.py#L36)。蒸馏相关字段：`messages`（全量历史，含
human/ai/tool）、`retrieved_frames/slides/transcripts`、`retrieval_plan`、
`grounding_report`、`agent_terminated`。**系统 prompt 不在 `messages` 里**。

### 2.4 Harness 预测路径（trajectory 落盘的挂钩点）

`_run_local_predictions`（[eval_harness.py:1337](app/eval_harness.py#L1337)）：
- `state = await app_graph.ainvoke({...})`（[:1381](app/eval_harness.py#L1381)）→ **返回的 `state["messages"]` 就是完整轨迹**（`add_messages` reducer 累积，顺序完整）。
- 现仅提取聚合字段进 `EvalPrediction`（[:1406](app/eval_harness.py#L1406)）：`agent_actions`（工具名序列，[_agent_actions:1442](app/eval_harness.py#L1442)）、`answer`、`retrieved_*`、`grounding_report`、`evidence_sufficiency`（已含 `agent_terminated`）。
- **没存 `messages`**——这是 Phase A 唯一要补的核心。

prediction cache key（[:1367](app/eval_harness.py#L1367)）= `case_id + model(orch|vlm) + prompt_fingerprint + video_id + AGENT_CODE_VERSION`。
`prompt_fingerprint()`（[eval_fingerprint.py:16](app/eval_fingerprint.py#L16)）hash 了
`QA_SYSTEM_PROMPT + _orchestrator_prompt(有/无视频) + AGENT_CODE_VERSION`。

### 2.5 评分 = 奖励信号（RL 阶段要重读）

`evaluate_case`（[eval_harness.py:147](app/eval_harness.py#L147)）是 **LLM judge 软门**：
- judge 判对 → `answer.passed = hallucination_free and uncertainty_ok`（**citation/keyword 软豁免**），且 retrieval/agent gate 也软豁免（[:212-221](app/eval_harness.py#L212)）。
- judge 判错 → 回退严格门：`answered(keyword) and citation_correct and halluc and uncertainty`。
- `passed = retrieval_ok and answer.passed and agent_ok`。

含义：**outcome > process**。对 SFT 没问题（Tier-1 只取干净正例）；对 RL 是隐患——
奖励几乎只看 final answer，会**欠约束工具使用**，策略可能学会绕过工具直接讨好 judge（§8）。

---

## 3. 领域决策：用 audiovisual 混合集（不用 NExT-GQA）

### 3.1 为什么混合集更对

- **工具覆盖**：50 个 smoke case 就出现 **12 种工具**作为动作，差异化工具全在场
  （`align_audiovisual_evidence`×24、`retrieve_slide_evidence`×11、`build_timeline`×3、
  `stitched_verify`×4、`retrieve_hypothesis_evidence`×8）。NExT-GQA 纯视觉，
  transcript/slide/align 这几支会在训练数据里**缺席**——等于丢掉项目最有价值的部分。
- **teacher 质量**：audiovisual smoke `pass_rate=0.90`（judge 对率高）→ Tier-1 产出高；
  NExT-GQA 仅 0.35–0.70 → 正例少、Tier-1 更少。
- **baseline 可比性**：Video-MME 是公开基准子集，"vs DeepSeek-v4-flash baseline" 仍成立，
  简历框架从 "NExT-GQA" 换成 "Video-MME 音视频子集" 反而更硬（多模态、更难）。

### 3.2 数据集现状

- `eval/audiovisual/questions.ready.jsonl`：**199 case / 59 视频**（已剔除坏真值 vme-377-2）。
  modality：visual 135 / joint 46 / overview 10 / audio 8。
- `eval/audiovisual/questions.jsonl`：203 原始溯源，**勿动**。
- 每 case 字段（[build_videomme_eval.py](scripts/build_videomme_eval.py) 产出）：`question_id`、
  `video_id`、`question`（含 `Candidates:`）、`expected_keywords`、`expected_citation_kinds`、
  `reference_answer`、`source_meta{youtube_id, correct_letter, options}`。

### 3.3 数据量是硬约束（用实测估算）

- mean **8.1** 步/case（p50=8, max=11）。
- 199 × 0.90 judge 对 ≈ 179 条正确轨迹 → Tier-1 过滤（去 guard 触发，粗估留 65–75%）≈ **120–135 条**。
- 每条切 ~7–8 样本 → **~900 样本**；再扣按视频留出的 eval split → 训练池 **~600–700 样本**。
- 对比那份 4500 train + 500 val 的目标 → **差约 5–7×**，且多样性被 **59 个视频**卡死（过拟合风险）。

**结论**：199-case 只够做 **pilot**（跑通管线、量真实 Tier-1 产出率），**不够**当最终 SFT。
补缺口的手段（按性价比）：
1. **扩 Video-MME 构建**（最干净）：源有上千题/数百视频，已有
   [build_videomme_eval.py](scripts/build_videomme_eval.py)。扩到 ~800–1000 题/~250 视频，
   量与工具覆盖都够。代价：新视频 ingest（ASR + slide OCR ~130s/视频 + 帧索引），~250 视频是数小时预处理。
2. **per-case 多采样增广**：teacher 升温/多 seed，每 case 取多条正确轨迹，做 2–3× 乘数（别当唯一来源）。
3. **混入 NExT-GQA 补量**：保留 audiovisual 做工具覆盖，NExT-GQA 补视觉 grounding 的量。
4. **下调数据目标**：14 工具的窄策略，~1000–1500 高质样本 + 3 epoch 也能出信号——但仅适合 pilot 验证，不作最终结论。

### 3.4 防数据污染（方法学红线）

teacher 轨迹采集集 与 Phase D 评测集 **必须按视频切分**（同一视频的多道题不得跨 train/eval）。
建议：59 视频 → ~47 train / ~12 eval。baseline（DeepSeek/doubao）也只在同一 held-out 视频集上跑。
否则 Phase D 的 pass_rate 与 "vs baseline" 全部失效。

---

## 4. Phase A：采集 teacher 轨迹

### 4.1 Task A.1 — 扩 harness 落盘完整 trajectory（核心工程）

在 [_run_local_predictions](app/eval_harness.py#L1337) 加 `--save-full-trajectory`（默认关，向后兼容）。
`state["messages"]` 已在手（[:1401](app/eval_harness.py#L1401)），需新增：

1. **序列化 messages**：每条 message → `{role, content, tool_calls?, tool_call_id?, name?, id?}`。
   role 映射：`human→user`、`ai→assistant`、`tool→tool`、`system`（见下）。
2. **补系统 prompt**：messages 里**没有** system。落盘时记录本 case 用的
   `_orchestrator_prompt(has_video=True, profile=<compute_video_profile(video_id)>)`
   渲染字符串（profile 影响内容，见 [graph.py:442-459](app/graph.py#L442)）。存为
   `system_prompt` 字段或直接作为 `messages[0]`。
3. **标记 guards_triggered**：从 message 流后处理推断（或在 graph 里埋点，二选一）：
   - `cap`/`empty` ← `state["agent_terminated"]`（已有）。
   - `forced_answer`/`forced_verify` ← 存在 `id ∈ {force_answer_with_evidence, force_verify_grounding}` 的 AIMessage。
   - `dedup` ← ToolMessage 内容与同签名早前 ToolMessage 重复（参考 [_dedup_tool_calls](app/graph.py#L210) 的签名逻辑）。
   - `stall` ← 末尾 AIMessage.content 等于某条 `answer_with_evidence` draft 且此前有 ≥2 条相同 verify 答案。
4. **扩 `EvalPrediction` + cache 序列化**：给 `prediction_to_dict/from_dict`（[:984](app/eval_harness.py#L984)、[:996](app/eval_harness.py#L996)）加可选 `messages`/`system_prompt`/`guards_triggered` 字段，**旧 cache 缺字段时降级为空**（向后兼容）。

落盘文件：`data/distillation/raw_trajectories/<run>.jsonl`，每行一个 case：

```json
{
  "case_id": "vme-303-1",
  "video_id": "9f87bb9bef0994eb",
  "question": "...",
  "system_prompt": "<_orchestrator_prompt 渲染>",
  "messages": [ {"role":"user","content":"..."},
                {"role":"assistant","content":"PLAN...","tool_calls":[...]},
                {"role":"tool","tool_call_id":"...","name":"retrieve_video_evidence","content":"<json,已截断base64>"},
                ... ],
  "guards_triggered": ["forced_verify"],
  "agent_terminated": null,
  "judge_correct": true,
  "judge_reason": "...",
  "pass_components": {"retrieval":..., "answer":..., "agent":...},
  "orchestrator_model": "deepseek-...",
  "vlm_model": "Qwen/Qwen3.5-...",
  "duration_sec": 12.3
}
```

> ⚠️ `judge_correct`/`judge_reason`/`pass_components` 来自 `evaluate_case`，
> 需把 judge 结果一并写入（harness 在 `--judge` 下已算，透传即可）。

### 4.2 Task A.2 — 跑 teacher

用 README 当前推荐的 **DeepSeek-v4-flash / doubao orchestrator + Qwen3.5 VLM** 配置（`.env`）：

```bash
conda run -n mbe-phase2 python scripts/eval_harness.py \
  --cases eval/audiovisual/questions.ready.jsonl \
  --judge \
  --prediction-cache data/eval/prediction_cache.jsonl \
  --output data/distillation/raw_trajectories/teacher_$(date +%Y%m%d).json \
  --save-full-trajectory
```

注意：
- **复用现有 prediction_cache**，已有结果不重跑（cache key 见 §2.4）。但**首次开
  `--save-full-trajectory` 时旧 cache 没有 messages** → 这些 case 需重跑或单独补采。
  建议：pilot 阶段用一个独立 cache 文件 `data/distillation/pred_cache_distill.jsonl` 全量重采。
- 支持断点续跑（cache 命中即跳）。
- pilot（199 case）预算极小；扩到 1000 case 时 watchdog：DeepSeek-v4-flash 估 ¥20–50，异常飙升立即停。

### 4.3 验收（写 `scripts/distill_validate_phase_a.py`）

- [ ] `raw_trajectories/` ≥ 应采 case 数的 90%
- [ ] 每条含 `messages` + `system_prompt` + `guards_triggered` + `judge_correct`
- [ ] 统计：judge 对率、各 guard 触发分布、mean tool_calls/case、mean messages 长度、**真实 Tier-1 产出率**（这是决定要不要立刻扩数据的关键数字）
- [ ] 输出 `data/distillation/phase_a_report.md`

---

## 5. Phase B：过滤 + 样本化

输出 `data/distillation/train.jsonl` / `val.jsonl`（按 §3.4 视频切分，不是随机切）。

### 5.1 Tier 过滤（`scripts/distill_filter_data.py`）

```python
RUNTIME_GUARDS = {"dedup", "stall", "salvage", "cap", "empty"}  # 项目实际 guard 名

def classify(traj) -> str:
    if not traj["judge_correct"]:                 return "tier_3"   # 全弃（teacher 自己错）
    if traj.get("agent_terminated") in ("cap","empty"): return "tier_3"
    g = set(traj.get("guards_triggered", []))
    if g & RUNTIME_GUARDS:                         return "tier_2"   # 默认弃，--include-tier-2 可纳
    # 注意:forced_answer/forced_verify 不算"脏",它们是正常 STOP DISCIPLINE 流程的一部分,
    # 但对应的 forced AIMessage 仍要在 §5.3 里排除为 target。
    return "tier_1"
```

默认只用 Tier-1。若 Tier-1 < 目标量（pilot 几乎必然），启用 `--include-tier-2` 并人工抽查 30 条。

### 5.2 样本化（一条轨迹切多个样本）

每个 **teacher 模型生成的** assistant message 作为一个样本的 target，其前缀作为 input。
input 必须复现**模型当时真正看到的上下文**：

```python
def slice_trajectory(traj):
    samples = []
    sys = traj["system_prompt"]            # §2.3:不在 messages 里,必须补
    msgs = traj["messages"]
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":       continue
        if is_forced_or_guard(m):          continue   # §5.3
        prefix = visible_filter(msgs[:i])  # §5.4:复现 _visible_messages 行为
        samples.append({"messages": [{"role":"system","content":sys}, *prefix, m],
                        "tools": load_tool_schemas()})
    return samples
```

### 5.3 排除非 teacher-策略的 target（项目特有，关键）

下列 assistant message **不可作 target**（它们是 harness 注入，不是模型决策）：
- `id ∈ {force_answer_with_evidence, force_verify_grounding}` 的 forced call。
- salvage/stall 产生的"末尾 draft 复制" AIMessage。
- dedup 合成的 ToolMessage（本就不是 assistant，但要确保 input 里它的位置与真实一致）。

判定写进 `is_forced_or_guard(m)`，并写单测覆盖。

### 5.4 复现 `_visible_messages`（input 侧保真）

orchestrator 实际看到的不是原始 `msgs[:i]`，而是经
[_visible_messages](app/graph.py#L701) 过滤：**剥掉历史轮 assistant 的 tool_calls**、
丢弃上一个 human 之前的非 human / 无 content 的 ai。eval 是单轮（一个 HumanMessage），
所以同一 case 内 `msgs[:i]` 基本等价于模型所见，**唯一必做的是补 system prompt**；
但为稳妥，过滤函数仍按 `_visible_messages` 同逻辑实现，并写单测对齐。

### 5.5 截断 base64（必做，否则 OOM）

`scripts/distill_truncate.py` 提供 `truncate_tool_result(name, content) -> content`：
- 任何 tool_result 里的图像 base64（`image_b64` 等）→ 替换为
  `"<frame_b64_omitted: t={timestamp}, scene={scene_id}>"`，**保留所有 metadata**
  （时间戳/scene_id/caption/score/marker）。
- `answer_with_evidence` / `verify_grounding` 的 result **完整保留**（本就无图）。
- 注意：当前多数检索 tool 的 ToolMessage 用 `_public_frame_refs`（已是 ref 非 base64），
  但仍要兜底扫描 `image_b64` 字段。**先写单测确认截断后 JSON 仍合法、metadata 不丢。**

### 5.6 工具 schema 导出（`data/distillation/tool_schemas.json`）

从 `app/tools.py:TOOLS` 经 `bind_tools` 同源导出 14 个工具的 OpenAI function schema，
**所有样本共享同一份**。版本与 `AGENT_CODE_VERSION` 一起记录。

### 5.7 输出格式（sharegpt，喂 LLaMA Factory）

每行一个样本：`{"messages":[{role:system},{role:user},{role:assistant,tool_calls},{role:tool},...,{role:assistant}], "tools":[<14 schema>]}`。
- 最后一个 assistant message 是 target，前面是 input；**只对最后一个 assistant 算 loss**。
- 用 transformers 的 `apply_chat_template`（template=qwen）渲染，**不要手写模板**。

### 5.8 验收（`scripts/distill_validate_phase_b.py`）

- [ ] train/val 行数达标（pilot 下据实记录，不强求 4500/500）
- [ ] 随机抽 20 样本人工 review：target 是合法 assistant message、非 forced；base64 已截断且 metadata 全；schema 与 TOOLS 一致
- [ ] 统计：target 类型分布（tool_call vs final_answer）、**14 个工具各被当 target 的次数**（确认覆盖，尤其 slide/align/timeline）、max_seq_len 分布（qwen tokenizer）
- [ ] 三个核心模块（filter/truncate/format）各有 pytest（[tests/test_distill_*.py](tests/)）
- [ ] 输出 `data/distillation/phase_b_report.md`

---

## 6. Phase C：SFT 训练（GPU server）

> 训练侧配置沿用 `orchestrator_distillation_spec.md` 中已针对 **4×RTX 3080 20GB（PCIe，无 NVLink）**
> 调过的数值，此处不改，仅做要点对齐。详细 yaml/显存账本/单卡 vs 多卡见那份 §4。

要点：
- LLaMA Factory + LoRA on `Qwen2.5-7B-Instruct`，`lora_rank=32, alpha=64`，`bf16`（Ampere 用 bf16 不用 fp16）。
- 20GB 单卡约束：`gradient_checkpointing=true`、`per_device_train_batch_size=1` + `gradient_accumulation_steps=32`、`cutoff_len=6144`（按 §5.8 的真实 token 分布微调；OOM 则降 4096）。
- **`cutoff_len` 必须先用 qwen tokenizer 统计本项目样本 token 分布再定**：audiovisual 轨迹
  mean 8.1 步、含截断后的 tool_result，序列偏长，先量 p95/p99/max 再拍板。
- `template: qwen`，sharegpt 格式，`columns: {messages, tools}`，`train_on_prompt=false`（只在 assistant token 算 loss）。
- 推荐方案 A：单卡训练 + 余卡跑 §7 eval；方案 B：4 卡 DDP（数据并行，单卡仍需放下整模型）。

验收：无 OOM/NaN；eval loss 持续下降并在 epoch 2–3 收敛；checkpoint 含 `adapter_model.safetensors`；smoke 推理能产出一个格式合法的 tool_call。

---

## 7. Phase D：部署 + 评测

### 7.1 vLLM serve（OpenAI 兼容 + LoRA + tool parser）

```bash
CUDA_VISIBLE_DEVICES=3 vllm serve /path/to/Qwen2.5-7B-Instruct \
  --enable-lora --lora-modules mrbigeye_orch=/path/to/checkpoints/mrbigeye_orch_v1 \
  --host 0.0.0.0 --port 8001 \
  --max-model-len 6144 --gpu-memory-utilization 0.90 --dtype bfloat16 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

7B bf16 ≈14GB，单卡 20GB 放得下 + KV cache；eval 串行无需高并发。**不要开 tensor-parallel**
（无 NVLink，PCIe 上 TP 反而拖慢）。tool parser 用 `hermes`（Qwen2.5）；不稳可换 SGLang（你已调过）。

### 7.2 切到本地模型（`.env.distill`）

```bash
ORCHESTRATOR_API_BASE_URL=http://<server-ip>:8001/v1
ORCHESTRATOR_API_KEY=EMPTY
ORCHESTRATOR_MODEL_NAME=mrbigeye_orch     # = lora-modules 名
ORCHESTRATOR_TEMPERATURE=0.2
ORCHESTRATOR_MAX_TOOL_CALLS=8
# VLM 保持与 teacher 同款（Qwen3.5），保证只对比 orchestrator 这一变量
```

切换不需手动 bump 缓存：换 `ORCHESTRATOR_MODEL_NAME` 会进 cache key（[§2.4](#24-harness-预测路径trajectory-落盘的挂钩点)）。
但**务必用独立 output 路径**，且评测集是 §3.4 的 **held-out 视频集**（不能含训练用过的视频）。

### 7.3 对比评测

```bash
# 学生(distill)在 held-out 集上，多 seed
for seed in 0 1 2; do
  conda run -n mbe-phase2 python scripts/eval_harness.py \
    --cases data/distillation/eval_heldout.jsonl \
    --sample-seed $seed --judge \
    --output data/eval/runs/heldout_$(date +%Y%m%d)_distill_seed${seed}.json
done
# baseline: 同 held-out 集、同 seed，换回 DeepSeek/doubao orchestrator 跑一遍
```

写 `scripts/distill_compare_runs.py`：输出 pass_rate(mean±std across seeds)、delta、
tool_calls/case、cost/case（学生用 vLLM token 估、teacher 从 `runs_index.csv` 估）、
guard 触发分布对比 → `data/distillation/phase_d_report.md`。

### 7.4 验收

- [ ] 学生 pass_rate 与 baseline 差距 ≤ 5pp（单 seed）或 ≤ 3pp（3-seed 均值）
- [ ] 单 case orchestrator 成本下降 ≥ 80%
- [ ] 任一 guard 触发率不显著高于 baseline（≤ baseline 的 2×）
- [ ] tool_calls/case 与 baseline 持平或更低

未达标 → Phase E；差距 >10pp → 回 Phase B 查数据（多半是 truncation 丢了 metadata 或样本量不足）。

---

## 8. Phase E（可选）：RL / DPO

仅当 Phase D 未达标时做。

### 8.1 ⚠️ 奖励信号陷阱（agentic RL 必读）

§2.5 已述：当前 `evaluate_case` 是 judge 软门，**judge 判对就软豁免 tool-use/retrieval/citation gate**。
直接拿 `passed` 当 RL reward → **outcome-only**，会欠约束工具使用，策略可能学会
"少调甚至不调工具、直接编一个讨好 judge 的答案"。做 RL 前需要：
- 设计**过程感知奖励**：把"正确的工具路由 / 证据 grounding / 引用正确"计入 reward（用
  `grounding_report`、`retrieval` gate、`agent_loop` 的**严格**信号，而非软豁免后的）。
- 或至少对"无工具调用就出答案"的轨迹给负奖励。

### 8.2 DPO（若走偏好对齐）

- chosen = Tier-1 的 assistant step；rejected = 同 question_type/同 step 的 Tier-3 对应 step。
- 用 BGE embed 找相似 question 配对（项目已有 bge-m3，见 [config.py:45](app/config.py#L45)）。
- 基于 SFT checkpoint 继续，`pref_beta=0.1`、`lr=5e-6`、1 epoch。目标 1000–2000 pairs。

---

## 9. 风险与对策

| 风险 | 触发 | 对策 |
|---|---|---|
| Tier-1 太少 | pilot 几乎必然（§3.3） | 先 pilot 验证管线；并行扩 Video-MME；或纳 Tier-2 + 人工抽查 |
| 数据污染 | train/eval 视频重叠 | **按视频切分**（§3.4），baseline 同 held-out 集 |
| forced-call 被当 target | 样本化未排除 | §5.3 的 `is_forced_or_guard` + 单测 |
| 系统 prompt 缺失 | 直接 `msgs[:i]` | §5.2 补 `system_prompt`（profile 相关） |
| OOM | 20GB 放不下 | cutoff 6144→4096；确认 gradient_checkpointing；rank 32→16；ZeRO-2 |
| tool parser 不稳 | vLLM 解析失败 | 换 hermes / SGLang |
| RL 退化 | 软门 reward | §8.1 过程感知奖励 |
| 缓存串味 | 改了 orchestrator 行为没失效 | bump `AGENT_CODE_VERSION`（[eval_fingerprint.py:13](app/eval_fingerprint.py#L13)）；改 prompt 自动失效 |

---

## 10. 交付物清单

```
Mr-Big-Eye/
├── app/eval_harness.py                      # 改:+ --save-full-trajectory / messages / guards 落盘 + cache 向后兼容
├── scripts/eval_harness.py                  # 改:透传新 flag
├── scripts/distill_filter_data.py           # 新:Tier 分级
├── scripts/distill_truncate.py              # 新:base64 截断
├── scripts/distill_train.yaml               # 新:LLaMA Factory 配置(§6)
├── scripts/distill_compare_runs.py          # 新:学生 vs baseline
├── scripts/distill_validate_phase_a.py      # 新
├── scripts/distill_validate_phase_b.py      # 新
├── scripts/distill_build_dpo_pairs.py       # 新(Phase E)
├── data/distillation/
│   ├── raw_trajectories/*.jsonl             # Phase A
│   ├── train.jsonl / val.jsonl              # Phase B(按视频切)
│   ├── eval_heldout.jsonl                   # held-out 视频集(Phase D)
│   ├── tool_schemas.json
│   └── phase_{a,b,d}_report.md
├── tests/test_distill_{filter,truncate,format}.py   # 新:三个核心模块单测
└── distillation_spec.md                     # 本文
```

---

## 11. 给 Claude Code 的执行提示

1. **严格按 Phase 顺序**，每 Phase 跑通验收再进下一个；先 pilot(199) 跑通 A→D，再扩数据。
2. **先写测试**：filter / truncate / format 三个核心模块必须 pytest。
3. **保护现有数据**：新文件进 `data/distillation/`，`data/eval/` 与
   `eval/audiovisual/questions.jsonl` 只读。
4. **禁改 `app/tools.py:TOOLS`**：schema 是 ground truth，改了等于训练数据作废。
5. **遇模糊决策**（如 Tier-2 是否纳入）：先用默认配置跑通整条 pipeline，再用 ablation 验证。
6. **训练在 GPU server**：数据准备/评测在本地，SSH + rsync 同步；§6 数值以 server 实测为准。
7. **保留所有中间产物**，Phase 间 debug 需回溯。
8. **本项目对话/文档用中文**，代码/标识符/路径/英文术语保持原文。
