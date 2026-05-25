# Mr. Big-Eye

> 面向长视频理解的浏览器端问答系统：离线把视频压成可检索索引，在线由 LangGraph
> agent 规划多步检索/扩证/假设/验证，最终生成带 `[FRAME:t=...]` 引用的回答。
> 业务侧零 VLM 权重，VLM 推理全部走云端 OpenAI-Compatible API；本地常驻只有
> BGE-M3（文本→场景召回）与 SigLIP2（文本→关键帧精排）两个轻量编码器。

## 核心定位

把"看长视频→回答任意问题"拆成 **预处理 / 检索 / 推理 / 评测** 四条独立可演进的流水线，并且每条流水线都有可量化的指标和可替换的 provider。结果：

| 维度 | 实现 |
| --- | --- |
| 视频压缩 | PySceneDetect 切场景 + decord 稠密抽帧 → 双索引（Caption + Frame） |
| 检索 | BGE-M3 caption→scene 召回 + SigLIP2 frame 精排，**配额混合**抗 BGE 路由失败 |
| 推理 | LangGraph 9 工具 agent loop：query planner → retrieve → assess → expand/timeline/hypothesis → answer → verify |
| 鲁棒性 | 工具调用去重、verify-stall 短路、空回复 salvage、tool-call 上限兜底 |
| 评测 | 规则指标 + LLM judge + **Soft-Waive**（结果对即过，路径对错只留 forensic 字段）；prompt+code fingerprint 自动失效预测缓存 |
| 可重复 | 每次 run 写 `run_meta` 块 + 追加 `runs_index.csv`，VLM/orchestrator 互换可量化对比 |
| 记忆 | LangGraph SQLite checkpointer（thread 级会话状态） + LangMem SQLite store（user 级长期记忆） |

## NExT-GQA 当前成绩

NExT-GQA 20-case 固定抽样（`--sample 20 --sample-seed 0`），同一 fingerprint 下交叉对比不同 VLM × Orchestrator 组合（来自 `data/eval/runs_index.csv`）：

| # | VLM | Orchestrator | Pass | Judge | Recall@K | TS Dist | 备注 |
|:-:|:----|:-------------|:----:|:-----:|:--------:|:-------:|:-----|
| 1 | MiMo-v2.5 | GLM-4.7-flash | 0.55 | 0.50 | 0.625 | 2.85s | v4 baseline |
| 2 | Doubao-Seed-Lite | GLM-4.7-flash | 0.40 | 0.40 | 0.700 | 1.91s | VLM 偏弱 |
| 3 | Qwen3-VL-30B-A3B | GLM-4.7-flash | 0.35 | 0.35 | 0.525 | 3.42s | orchestrator 偏弱 |
| 4 | Qwen3-VL-30B-A3B | GLM-4.7-flash | 0.45 | 0.40 | 0.700 | 2.67s | v5 prompt+guard 修复 |
| **5** | **Qwen3-VL-30B-A3B** | **Doubao-Seed-Pro** | **0.70** | **0.70** | **0.800** | **2.08s** | **当前最佳** |
| 6 | Qwen3-VL-30B-A3B | *(无检索控制组：均匀抽 6 帧 + VLM)* | 0.70 | 0.70 | 0.675 | 2.01s | 检索召回 +18%，但 VLM 上限把 pass 锁住了 |

可读出的结论：

- **Orchestrator 比 VLM 更敏感**：同一 VLM，把 GLM-4.7-flash 换 Doubao-Pro，pass_rate 0.45 → 0.70。
- **Pass == Judge**：Soft-Waive 让最终判分完全由 LLM judge 主导，规则 gate 只在 judge 错杀时降权。
- **检索召回 0.80 vs 0.675**：检索栈把对的帧多送了 18%，但在这 20 题样本上 VLM 视觉识别成了瓶颈——下一步重点是 LensWalk-inspired 的 multi-segment Observer 工具，详见 [paper_references/paper_reference_plan.md](paper_references/paper_reference_plan.md)。

---

## 架构总览

```mermaid
flowchart TB
    UI["Browser UI (app/static)"] --> API["FastAPI (app/main.py)"]
    API --> UPLOAD["POST /upload + SSE /api/preprocess_stream"]
    API --> CHAT_STREAM["GET /api/chat_stream (登录会话, SSE)"]
    API --> CHAT_DIRECT["POST /chat (匿名直答)"]
    API --> SESSION["sessions / videos / login API"]
    API --> DB[("SQLite: users / sessions / videos")]

    UPLOAD --> PRE["preprocess_video (app/preprocess.py)"]
    PRE --> PROBE["decord probe + PySceneDetect"]
    PROBE --> SCENE_FRAMES["frames_scene/ (场景中间帧)"]
    PROBE --> DENSE_FRAMES["frames_dense/ (DENSE_FPS 抽帧)"]
    SCENE_FRAMES --> CAPTION["VLM API: scene caption"]
    CAPTION --> BGE_BUILD["BGE-M3 caption embeddings"]
    DENSE_FRAMES --> SIG_BUILD["SigLIP2 image embeddings"]
    BGE_BUILD --> CIDX[("Chroma: caption_index")]
    SIG_BUILD --> FIDX[("Chroma: frame_index")]
    PRE --> CACHE[("data/cache/<video_id>/")]

    CHAT_STREAM --> GRAPH["LangGraph orchestrator (app/graph.py)"]
    GRAPH --> ORCH_LLM["Orchestrator LLM (OpenAI-compatible chat)"]
    GRAPH --> TOOLS["9 evidence tools (app/tools.py)"]
    TOOLS --> RET["two_stage_retrieve (app/retrieval.py)"]
    RET --> CIDX
    RET --> FIDX
    TOOLS --> VQA["VLM client (app/vqa.py)"]
    VQA --> REMOTE["Doubao Responses / OpenAI-compat Chat"]
    GRAPH --> CKPT[("SQLite: graph_checkpoints")]
    GRAPH --> MEM["LangMem manager (app/memory.py)"]
    MEM --> LM[("SQLite: langmem_store")]
    CHAT_DIRECT --> RET
    CHAT_DIRECT --> VQA
```

---

## 技术深度与亮点

### 1. 两阶段检索 + 配额混合（抗 BGE 路由失败）

#### 1.1 离线索引构建

```mermaid
flowchart LR
    V[上传 MP4] --> HASH[SHA256 → video_id]
    HASH --> PROBE[decord probe]
    PROBE --> SCENE[PySceneDetect<br/>threshold=27.0]
    SCENE --> MID[场景中间帧<br/>frames_scene/]
    MID --> CAP[VLM API<br/>scene caption]
    CAP --> BGEIDX[BGE-M3<br/>caption_index]
    PROBE --> DENSE[按 DENSE_FPS<br/>稠密抽帧]
    DENSE --> SIGIDX[SigLIP2<br/>frame_index]
    BGEIDX --> DONE[set_video_status=done]
    SIGIDX --> DONE
```

#### 1.2 在线检索（配额混合是关键）

```mermaid
flowchart LR
    Q[用户问题] --> BGEQ[BGE-M3 文本向量]
    BGEQ --> CIDX[(Caption Index)]
    CIDX --> SCENES[Top-N 场景<br/>+ 时间区间]
    Q --> SIGQ[SigLIP2 文本向量]
    SIGQ --> SCENED[Scene-Gated 检索<br/>quota = 2/3 · K]
    SIGQ --> GLOBAL[全局 Un-Gated 检索<br/>quota = 1/3 · K]
    SCENES --> SCENED
    SCENED --> MERGE{合并去重}
    GLOBAL --> MERGE
    MERGE --> TOPK[Top-K 关键帧<br/>送给 Agent]
```

**配额混合的动机**：BGE caption→scene 是有失败率的（场景文字描述 ≠ 用户语义问题）。如果 frame 检索完全被 BGE 选中的场景时间窗 gate 住，BGE 一旦路由错，整轮就死。代码在 [app/retrieval.py:84-111](app/retrieval.py#L84-L111)：默认把 2/3 配额留给 scene-gated SigLIP（精度），剩下 1/3 留给全局 un-gated SigLIP（召回兜底）。总帧数和 VLM token 成本不变。

#### 1.3 缓存目录契约

```
data/cache/<video_id>/
├── meta.json              # duration / fps / scene 列表
├── frames_scene/          # 场景中间帧
├── frames_dense/          # tNNNN.N.jpg，按 DENSE_FPS
├── captions.jsonl         # 每场景 caption + meta
├── caption_index/         # Chroma persistent
├── frame_index/           # Chroma persistent
└── .done                  # set_video_status 写入
```

`POST /upload` 用 SHA256 算 `video_id`，相同视频复传秒级命中缓存。

---

### 2. LangGraph Agent Loop with 9-Tool Toolbox

#### 2.1 控制流

```mermaid
flowchart TB
    START([START]) --> ORCH[orchestrator<br/>LLM with bind_tools]
    ORCH -->|tools_condition=tools| TOOL[tool_node<br/>ToolNode TOOLS]
    ORCH -->|no tool_calls| MEM[memory_write_node]
    TOOL --> ORCH
    MEM --> END([END])

    ORCH -.->|tool-call cap hit| SALVAGE[_salvage_draft_answer]
    SALVAGE -.-> ORCH
    ORCH -.->|verify_grounding 两次同答| STALL_BREAK[_verify_grounding_stalled<br/>→ emit draft]
    STALL_BREAK -.-> ORCH
```

#### 2.2 工具清单（[app/tools.py](app/tools.py)）

| 工具 | 作用 | 何时调用 |
| --- | --- | --- |
| `retrieve_video_evidence` | 按 question_type + retrieval_profile 检索证据帧 | 总是第一个 |
| `assess_evidence_sufficiency` | 给出 sufficient/insufficient + recommended_next_action | 检索后、回答前 |
| `build_timeline` | 围绕 retrieved_frames 生成时间线 | temporal_order / counting / comparison |
| `expand_temporal_evidence` | 围绕给定时间戳扩展前后窗口 | 证据局部稀疏时 |
| `retrieve_hypothesis_evidence` | 针对具体视觉假设做二次检索 | 候选答案要互相排除时 |
| `answer_with_evidence` | 仅基于当前 frame set 出 draft 答案 | 证据足够 |
| `verify_grounding` | 检查 `[FRAME:t=...]` 是否匹配已检索帧、visual claim 是否缺 citation、negative answer 是否限定作用域 | 出 draft 后 |
| `search_user_memories` | 读取 LangMem 中的 user 偏好/历史 | 与用户偏好相关时 |
| `multimodal_vqa` | （legacy 兼容快捷工具，prompt 中明确建议不优先用） | 极端 fallback |

#### 2.3 7 种 question_type × 6 种 retrieval_profile

Query planner 在 prompt 里强制 orchestrator 先分类、再选 profile。每个 profile 影响 `top_n_scenes` / `top_k_frames` 默认值：

- `focused`：单点细节（top_n=3, top_k=8）
- `balanced`：默认
- `broad`：综述/摘要（top_n=10, top_k=20）
- `temporal`：时序/计数/比较（top_n=8, top_k=18）
- `detail`：高分辨率视觉/OCR（top_k=18）
- `negative_check`：否定回答前的兜底扫荡（top_n=12, top_k=24）

详见 [app/tools.py](app/tools.py) `_profile_defaults` 与 `_resolve_retrieval_plan`。

#### 2.4 GraphState（持久化在 SQLite checkpointer）

```python
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    video_id: str | None
    user_id: str
    retrieved_frames: Annotated[list[dict], _last_write]
    retrieved_scene_hits: Annotated[list[dict], _last_write]
    retrieval_plan: Annotated[dict, _last_write]
    timeline: Annotated[list[dict], _last_write]
    hypotheses: Annotated[list[dict], _last_write]
    evidence_sufficiency: Annotated[dict, _last_write]
    draft_answer: Annotated[str, _last_write]
    grounding_report: Annotated[dict, _last_write]
    agent_terminated: str | None
```

`_last_write` 是自定义 reducer——一轮中如果 orchestrator 并发触发多个改同一字段的工具（LangGraph 多写冲突），保留最新非 None 的值，避免节点报错。

---

### 3. Orchestrator 鲁棒性护栏

实际 production 调用中，LLM 会产生各种"语法上合法但语义死循环"的输出。`app/graph.py` 加了 4 道护栏：

| 护栏 | 触发条件 | 行为 |
| --- | --- | --- |
| **工具调用去重** ([_dedup_tool_calls](app/graph.py#L182)) | 同一 `(name, args)` 签名在历史里出现过 | 合成 `ToolMessage`（用上次缓存内容），不再 fire 真实工具；如果全是 dup 且有 draft，直接 emit draft 终止 |
| **verify_grounding 停滞** ([_verify_grounding_stalled](app/graph.py#L244)) | 倒数两次 `verify_grounding` 返回相同 answer | 短路出 draft，不再调模型 |
| **空回复 salvage** ([_make_orchestrator](app/graph.py#L84)) | 模型返回 `tool_calls=[]` 且 `content=""` | 从历史里捞最长 non-truncated `answer_with_evidence` payload；若仍无，注入 coercion HumanMessage 重试一次 |
| **Tool-call 上限** ([orchestrator_max_tool_calls=8](app/config.py)) | 一轮工具调用数 ≥ 8 | salvage draft；无 draft 则置 `agent_terminated="cap"` |

每个护栏对应 [tests/test_graph_orchestrator.py](tests/test_graph_orchestrator.py) 里的一个回归测试。

---

### 4. Soft-Waive 评测哲学：结果对就过，路径对错留 forensic

传统 harness：retrieval / answer / agent_loop 三段都 strict gate，AND。问题：LLM judge 已经判答案正确，但 retrieval 召回因 tolerance 边缘 fail（如 ts_dist=2.1s vs tolerance=2.0s），整 case 被错杀。

[app/eval_harness.py:evaluate_case](app/eval_harness.py#L107) 的解法：当 `judge.correct=True` 时，**citation / agent / retrieval** 三个 gate 全部 soft-waive，并在 JSON 里留下 `*_soft_waived: True` 字段供 forensic：

```python
# 简化伪代码
if judge_pass:
    answer["citation_soft_waived"] = not answer["citation_correct"]
    answer["passed"] = hallucination_free and uncertainty_ok
    agent["soft_waived"] = (agent["passed"] is False)
    retrieval["soft_waived"] = (retrieval["passed"] is False)

retrieval_ok = retrieval["passed"] is not False or retrieval["soft_waived"]
agent_ok = agent["passed"] is not False or agent["soft_waived"]
passed = retrieval_ok and answer["passed"] and agent_ok
```

效果：当前 NExT-GQA 20-case run 的 `pass_rate == judge_correct_rate == 0.70`，三个 strict gate 在严格统计里仍可读（`agent_loop_pass_rate=0.65, retrieval_pass_rate=0.75`），但不会拖累 outcome 分。

---

### 5. Prediction Cache + Prompt-Aware Fingerprint

跑 harness 时直接调本地 agent 是 token-密集型操作。[app/eval_fingerprint.py](app/eval_fingerprint.py) 把 cache key 设计成：

```python
key = sha1(QA_SYSTEM_PROMPT ‖ _orchestrator_prompt(has_video=True) ‖
           _orchestrator_prompt(has_video=False) ‖ AGENT_CODE_VERSION)[:12]
```

含义：

- **改 prompt**：fingerprint 自动变，受影响的缓存条目失效，未变的继续命中。
- **改 agent 运行时行为**（dedup 逻辑、salvage 逻辑等不动 prompt 的改动）：手动 bump `AGENT_CODE_VERSION`（当前 `v5`），全量重跑。
- **改打分逻辑**（如 Phase B 的 soft citation gate）：完全不动 prediction，只在已有缓存上重算指标，秒级出报告。

`data/eval/prediction_cache.jsonl` 是 append-only JSONL（类似 [app/eval_harness.py](app/eval_harness.py) 的 JudgeCache 模式）。

---

### 6. 多 Provider VLM + Run-Level Footprint

#### 6.1 两种 wire format 自动切换

[app/vqa.py](app/vqa.py) 封装：

- `responses`：火山方舟 Doubao `POST /responses`，input 是 `[{role, content:[{type:input_image|input_text}]}]`
- `chat_completions`：OpenAI-compatible（ModelScope / 小米 MiMo / Volcengine /...），messages 是 `[{role, content:[{type:image_url|text}]}]`

切换只动 `.env`，业务零改动。

#### 6.2 每次 run 留可对比指纹

`scripts/eval_harness.py` 在每次 run 时往输出 JSON 写 `run_meta` 块，并 append 一行到 `data/eval/runs_index.csv`：

```
timestamp,vlm_model,orch_model,fingerprint,version,n,seed,pass_rate,
answer_pass_rate,judge_correct_rate,recall_mean,ts_dist_mean,output_path
```

VLM swap 体验：换 provider → 跑一次 → `column -t -s, data/eval/runs_index.csv` 直接对比。本 README 顶部的成绩表就是这个 CSV 导出的。

---

### 7. Grounding + Negative-Answer Protocol

回答里写 `[FRAME:t=29.7]` 这类 marker，前端把 marker 替换成内联缩略图。`verify_grounding` 工具做四件事：

1. marker 是否落在已检索帧附近（容忍 tolerance）。
2. visual claim 是否带 citation（如"红衣男子在跑步"但没引用任何帧）。
3. 否定回答（"没有/不存在/未看到/没看到"）是否先调用过 `negative_check` profile 检索。
4. 否定回答是否把作用域限定在"已检查证据/当前关键帧"而非全视频（防止过度泛化否定）。

效果：模型倾向回答"在已检查的关键帧中没有看到 X"而不是直接"视频中没有 X"。

---

### 8. 双层记忆系统

| 层 | 后端 | 作用域 | 内容 |
| --- | --- | --- | --- |
| LangGraph Checkpointer | `data/graph_checkpoints.sqlite3` | thread_id（一个对话 = 一个视频） | 完整 `GraphState`，支持断线重连/历史回看 |
| LangMem | `data/langmem_store.sqlite3` | user_id | 用户偏好、跨 session 的语义事实 |

[app/graph.py:_make_memory_write_node](app/graph.py#L377) 在每轮回答后，把最近 12 条可见消息送给 LangMem manager 抽取并 upsert。orchestrator 通过 `search_user_memories` 工具主动检索。两层互不串扰——视频内的临时实体不会污染跨 session 的 user memory。

---

### 9. 控制组：无检索基线

[scripts/eval_no_retrieval.py](scripts/eval_no_retrieval.py) 提供一个"均匀抽 N 帧 + 一次 VLM 调用"的 control group，输出 prediction JSONL 可直接喂回 `eval_harness.py --predictions`：

```bash
# 1. 跑控制组生成预测
python scripts/eval_no_retrieval.py \
    --cases tests/fixtures/eval_cases_nextgqa.jsonl \
    --output data/eval/preds_no_retrieval_$(date +%Y%m%d_%H%M).jsonl \
    --num-frames 6 --sample 20 --sample-seed 0

# 2. 用同一 harness 评分
python scripts/eval_harness.py \
    --cases tests/fixtures/eval_cases_nextgqa.jsonl \
    --predictions data/eval/preds_no_retrieval_*.jsonl \
    --output data/eval/runs/nextgqa_$(date +%Y%m%d_%H%M)_no_retrieval.json \
    --judge
```

意义：成绩表第 6 行表明在当前 20-case 样本上，**两阶段检索把召回从 0.675 抬到 0.800**（+18%），但 pass_rate 都是 0.70——VLM 视觉识别已经成了瓶颈。下一步收益不在 retrieval，在 [paper_reference_plan.md](paper_references/paper_reference_plan.md) 里的 multi-segment Observer 工具。

---

## 完整请求流程（提问到回答）

```mermaid
sequenceDiagram
    autonumber
    participant UI as Browser
    participant API as FastAPI
    participant G as LangGraph
    participant O as Orchestrator LLM
    participant T as Tools
    participant R as two_stage_retrieve
    participant V as VLM API
    participant LM as LangMem

    UI->>API: GET /api/chat_stream?session=X&q=...
    API->>G: ainvoke(state with video_id, user_id, q)
    G->>O: SystemPrompt + history
    O-->>G: tool_calls=[retrieve_video_evidence(...)]
    G->>T: ToolNode dispatch
    T->>R: two_stage_retrieve(video_id, q)
    R-->>T: frames, timestamps, scene_hits
    T-->>G: ToolMessage + state update
    G->>O: append ToolMessage
    O-->>G: tool_calls=[assess_evidence_sufficiency(...)]
    G->>T: dispatch
    T-->>G: {sufficient: false, recommended_next_action: "expand_temporal_evidence"}
    O-->>G: tool_calls=[expand_temporal_evidence(...)]
    G->>T: dispatch
    T->>R: dense frame window
    R-->>T: more frames
    T-->>G: state update
    O-->>G: tool_calls=[answer_with_evidence(...)]
    G->>T: dispatch
    T->>V: POST chat/completions (images + question)
    V-->>T: draft answer + [FRAME:t=...]
    T-->>G: ToolMessage(draft)
    O-->>G: tool_calls=[verify_grounding(draft)]
    G->>T: dispatch
    T-->>G: {grounded: true}
    O-->>G: AIMessage(content=draft, tool_calls=[])
    G->>LM: write_memories(visible messages)
    G-->>API: stream tokens
    API-->>UI: SSE chunks
```

虚线护栏（dedup / verify-stall / salvage / cap）随时可能介入并提前 emit 最终回答。

---

## 快速开始

### 环境

```bash
conda create -n mr-big-eye python=3.10 -y
conda activate mr-big-eye

# 可选：国内 PyPI 源
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple

pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 填 VLM_API_KEY / ORCHESTRATOR_API_KEY，详见 .env.example
```

ModelScope（推荐，国内免代理）+ Volcengine 组合：

```env
# VLM
VLM_API_PROVIDER=modelscope
VLM_API_FORMAT=chat_completions
VLM_API_BASE_URL=https://api-inference.modelscope.cn/v1
VLM_API_KEY=<MODELSCOPE_KEY>
VLM_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct

# Orchestrator
ORCHESTRATOR_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/
ORCHESTRATOR_API_KEY=<ARK_KEY>
ORCHESTRATOR_MODEL_NAME=doubao-seed-2-0-pro-260215
```

火山方舟单独跑 VLM：

```env
VLM_API_FORMAT=responses
VLM_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VLM_API_KEY=<ARK_KEY>
VLM_MODEL_NAME=doubao-seed-2-0-pro-260215
```

### 下载本地检索模型

```bash
python scripts/download_models.py     # BGE-M3 + SigLIP2，从 ModelScope CDN
```

### 启动

```bash
bash scripts/launch_app.sh            # http://localhost:8000
# 关停
lsof -ti :8000 | xargs -r kill
```

### Smoke Test

```bash
python scripts/smoke_test.py \
  --video tests/fixtures/short_clip.mp4 \
  --question "What object moves across the video?"
```

---

## API

| Endpoint | 作用 |
| --- | --- |
| `GET /` | 浏览器 UI |
| `POST /upload` | 上传视频 → 返回 `video_id` 与 SSE 预处理流 URL |
| `GET /status/{video_id}` | 查询预处理状态 |
| `GET /api/preprocess_stream/{video_id}` | SSE：预处理阶段进度 |
| `POST /chat` | 匿名直答（direct retrieval + VQA，不走 agent） |
| `GET /api/chat_stream` | 登录会话 SSE：走完整 LangGraph agent loop |
| `POST /api/login` | 本地 name-tag 登录 |
| `GET/POST/PATCH /api/sessions[/...]` | 会话 CRUD + 历史消息 |
| `GET /api/videos` | 用户已分析的视频列表 |

---

## 评测管线

### 数据集准备（NExT-GQA only；LongVideoBench 已 descope）

```bash
# 1. 拉数据 + 采样到 20 题
python scripts/eval_prepare_datasets.py --datasets nextgqa --sample 20

# 2. 转 EvalCase JSONL
python scripts/eval_convert_nextgqa.py \
  --raw data/eval/_raw/nextgqa \
  --output tests/fixtures/eval_cases_nextgqa.jsonl

# 3. 拉视频 + 走完整 preprocess 写缓存
python scripts/eval_ingest_videos.py \
  --cases tests/fixtures/eval_cases_nextgqa.jsonl
```

### 跑评测

```bash
# 直接调本地 agent（默认）
python scripts/eval_harness.py \
  --cases tests/fixtures/eval_cases_nextgqa.jsonl \
  --sample 20 --sample-seed 0 --judge \
  --prediction-cache data/eval/prediction_cache.jsonl \
  --output data/eval/runs/nextgqa_$(date +%Y%m%d_%H%M).json

# 用离线 prediction JSONL（控制组 / 其他系统对比）
python scripts/eval_harness.py \
  --cases tests/fixtures/eval_cases_nextgqa.jsonl \
  --predictions some_predictions.jsonl \
  --output data/eval/runs/<name>.json --judge
```

### Judge

`.env` 里 `JUDGE_API_KEY` 留空时，harness 会要求显式 `--no-judge` 或回退到 `VLM_API_*`（带"自评 bias" 警告）。建议生产路径填一个独立 judge：

```env
JUDGE_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
JUDGE_API_KEY=<ARK_KEY>
JUDGE_MODEL_NAME=doubao-seed-2-0-pro-260215
```

Judge 结果写入 `data/eval/judge_cache.jsonl`（key=case_id+judge_model+prompt_hash），改打分逻辑时无需重跑 VLM。

### 解读 run 报告

每次 `eval_harness.py` 输出：

- `data/eval/runs/<name>.json`：`run_meta` + `summary` + per-case `results`。
- `data/eval/runs/<name>.md`：人类可读 markdown 报告。
- `data/eval/runs_index.csv`：追加一行用于跨 run 对比。

---

## 配置

| 变量 | 作用 | 默认/示例 |
| --- | --- | --- |
| `VLM_API_FORMAT` | `responses` / `chat_completions` | `chat_completions` |
| `VLM_API_BASE_URL` | VLM 端点 | `https://api-inference.modelscope.cn/v1` |
| `VLM_API_KEY` | VLM Bearer | 空 |
| `VLM_MODEL_NAME` | caption + 多帧 QA 模型 | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| `VLM_API_TIMEOUT` | VLM HTTP 超时 | `120` |
| `ORCHESTRATOR_API_BASE_URL` | tool-call orchestrator 端点；空则复用 VLM | 空 |
| `ORCHESTRATOR_API_KEY` | orchestrator key | 空 |
| `ORCHESTRATOR_MODEL_NAME` | orchestrator 模型 | 空 |
| `ORCHESTRATOR_TEMPERATURE` | 默认 `0.2` |
| `ORCHESTRATOR_MAX_TOOL_CALLS` | 单轮工具上限 | `8` |
| `ORCHESTRATOR_STREAMING` | 某些 proxy 不支持工具流式，默认关 | `false` |
| `JUDGE_*` | LLM judge 独立端点，空则回退 VLM | 空 |
| `LANGMEM_*` | LangMem 抽取端点，空则回退 VLM | 空 |
| `LANGMEM_QUERY_LIMIT` | 单次 user memory 检索条数 | `6` |
| `TOP_N_SCENES` / `TOP_K_FRAMES` | 检索默认 | `5` / `12` |
| `PLANNER_MAX_TOP_N_SCENES` / `PLANNER_MAX_TOP_K_FRAMES` | planner 上限 | `12` / `36` |
| `VQA_MAX_FRAMES` | 实际送 VLM 的证据帧上限 | 本地 `6`，服务器 `12` |
| `VQA_MAX_IMAGE_SIDE` / `VQA_IMAGE_QUALITY` | 输入图片预处理 | `448` / `75` |
| `DENSE_FPS` | 稠密抽帧帧率 | `1.0` |
| `SCENE_DETECT_THRESHOLD` | PySceneDetect | `27.0` |
| `MAX_VIDEO_DURATION_SEC` | 最大视频时长 | `600` |
| `MODELS_DEVICE` | 本地 BGE/SigLIP 设备 | `cpu` / `cuda:0` |
| `LOAD_MODELS_ON_STARTUP` | 启动时加载本地编码器 | 本地 `false`，服务器 `true` |
| `DATA_DIR` | 运行数据目录 | `./data` |

服务器迁移见 `.env.server.example`。

---

## 目录结构

```text
.
├── app/
│   ├── main.py             # FastAPI 路由 + SSE + 后台预处理
│   ├── config.py           # Pydantic Settings
│   ├── vqa.py              # VLM API client（responses + chat_completions）
│   ├── models.py           # BGE-M3 / SigLIP2 wrappers + lazy load/release
│   ├── preprocess.py       # 视频预处理 + 索引构建 + set_video_status
│   ├── retrieval.py        # 两阶段检索 + 配额混合
│   ├── graph.py            # LangGraph orchestrator + 4 道护栏
│   ├── tools.py            # 9 工具实现 + planner 解析
│   ├── memory.py           # LangMem manager / store
│   ├── eval_harness.py     # 评测指标 + Soft-Waive + 缓存
│   ├── eval_fingerprint.py # AGENT_CODE_VERSION + prompt fingerprint
│   ├── eval_datasets.py    # 数据集元信息
│   ├── cache.py            # 文件缓存 + .done 状态
│   ├── db.py               # SQLite users/sessions/videos
│   ├── progress.py         # SSE pub/sub
│   ├── schemas.py / usernames.py
│   └── static/             # 浏览器 UI
├── scripts/
│   ├── launch_app.sh
│   ├── download_models.py
│   ├── smoke_test.py
│   ├── eval_harness.py
│   ├── eval_no_retrieval.py        # 控制组：均匀采 N 帧 + VLM
│   ├── eval_convert_nextgqa.py     # NExT-GQA → EvalCase JSONL
│   ├── eval_convert_longvideobench.py  # （legacy，LVB 已 descope）
│   ├── eval_prepare_datasets.py    # 拉数据集 + 采样
│   └── eval_ingest_videos.py       # 跑 preprocess + 绑定 video_id
├── tests/
│   ├── fixtures/
│   │   ├── short_clip.mp4
│   │   ├── eval_cases.jsonl
│   │   ├── eval_cases_nextgqa.jsonl
│   │   ├── eval_cases_nextgqa_smoke.jsonl
│   │   ├── eval_predictions.jsonl
│   │   └── datasets/
│   ├── test_eval_harness.py
│   ├── test_eval_converters.py
│   ├── test_graph_orchestrator.py
│   ├── test_main_stream_contract.py
│   ├── test_tools_planner.py
│   ├── test_retrieval.py
│   ├── test_vqa.py
│   ├── test_cache.py / test_db.py / test_memory.py / test_usernames.py
├── paper_references/
│   ├── LensWalk.pdf                # 下一阶段灵感来源
│   └── paper_reference_plan.md     # LensWalk-inspired 改造计划（5 个 phase）
├── .env.example / .env.server.example
└── requirements.txt
```

运行时目录：

```text
data/
├── uploads/
├── mr_big_eye.sqlite3                  # users/sessions/videos
├── graph_checkpoints.sqlite3           # LangGraph thread 状态
├── langmem_store.sqlite3               # LangMem user memory
├── cache/<video_id>/                   # 见 1.3
└── eval/
    ├── _raw/                           # 原始数据集
    ├── datasets/                       # 处理后样本
    ├── prediction_cache.jsonl          # case_id + fingerprint → prediction
    ├── judge_cache.jsonl               # case_id + judge_model → judge 结果
    ├── runs/<dataset>_<ts>_<tag>.json  # 每次 run 报告
    ├── runs/<dataset>_<ts>_<tag>.md
    └── runs_index.csv                  # 跨 run 对比
```

`data/` 与 `models/` 默认不入 Git。

---

## 测试

```bash
python -m pytest -q
```

当前覆盖：缓存 / DB / 用户名 / VQA payload / 两阶段检索契约 / LangGraph orchestrator 4 道护栏 / 工具 planner profile 映射 / evidence sufficiency / grounding report / Soft-Waive 评分 / 数据集 converter / prediction cache / `main.py` SSE stream contract。

---

## 已知限制

- 用户身份是本地 name-tag，不是认证系统。
- 预处理走 FastAPI 后台任务，不是生产级队列。
- 远程 VLM 延迟 / 并发 / 费用取决于 provider。
- `POST /chat` 匿名路径仍是 direct retrieval + VQA；只有登录 SSE 走完整 agent。
- Grounding 是规则型 + LLM judge 混合；细粒度语义对齐还可继续加强。
- caption_index / frame_index 没有版本号，升级 BGE / SigLIP 模型需手动清缓存。

## 后续方向

- **LensWalk-inspired agent 升级**（详见 [paper_references/paper_reference_plan.md](paper_references/paper_reference_plan.md)）：
  - `stitched_verify` 工具：跨多窗口对比，对应 temporal_order / comparison
  - `segment_focus` 工具：单窗口 1fps 密采，对应 visual_detail
  - VQA prompt LensWalk-style 严格 scope 约束（防 Evidence Dilution）
  - Orchestrator THINK→PLAN→OBSERVE 显式声明
  - Subject Registry：单视频跨 turn 实体追踪（GraphState 字段，per thread_id 持久化，**不**进 LangMem）
- 索引产物加版本号，模型升级自动失效缓存。
- 引入 Redis / Celery 等任务队列替代后台任务。
- 时间轴 UI：点击关键帧跳转视频时间点。

---

## Highlights (English TL;DR)

- Long-video QA via PySceneDetect + BGE-M3 caption retrieval + SigLIP2 frame reranking with **quota-blended fallback** against BGE routing failures.
- Remote VLM API only (Doubao Responses / OpenAI-compat Chat); zero local VLM serving stack.
- LangGraph agent loop with 9 tools, 4 production-grade guards (tool-call dedup, verify-grounding stall short-circuit, empty-content salvage, tool-call cap).
- **Soft-Waive scoring**: when LLM judge accepts, citation/agent/retrieval gates soft-waive but keep strict signal in JSON for forensics.
- **Prompt-aware prediction cache**: SHA1(prompts ‖ AGENT_CODE_VERSION) auto-invalidates affected entries on prompt edits.
- **Run-level footprint**: per-run `run_meta` + append-only `runs_index.csv` for reproducible VLM/orchestrator comparison.
- Two-layer memory: LangGraph SQLite checkpointer for thread state + LangMem SQLite store for cross-session user memory.
- Reproducible quality harness over NExT-GQA with a no-retrieval control-group baseline.
