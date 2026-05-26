# Mr-Big-Eye 音画 QA 升级路线图

> 目标：从纯视觉 VQA 智能体升级为能处理「视频 + 语音 + 板书/PPT」的实用音画 QA 系统，典型场景是课堂录像知识问答。
> 范围：不再追 NExT-GQA。所有评估指向真实音画用例。

## 当前位置

- [x] 视觉 RAG（CLIP 帧检索、scene 检索、Observer 子工具）
- [x] LangGraph orchestrator（Qwen3.5 / DeepSeek）+ VLM（Qwen2.5-VL 等）
- [x] FINAL ANSWER PROTOCOL + MCQ 硬规则 + SUBJECT_DELTAS 协议
- [x] 本地 ASR：SenseVoice-Small + FSMN-VAD 跑通（RTF 31.9× on RTX 4060）
- [ ] ASR 后处理 + 持久化
- [ ] 文本检索栈（FTS5 + bge-m3 hybrid）
- [ ] 音画联合证据对齐
- [ ] 课堂场景：板书/PPT OCR
- [ ] 真实音画评测集

---

## Handoff context for Codex (Phase 0)

Codex 拿不到此前 session 里建立的隐性 context；下面是实施每个 Phase 前必须知道的项目约束、关键文件、和决策门槛。**先读完这一节再动 Phase A**。

### 1. 环境

- Python 解释器固定走 `mbe-phase2` conda env：`/home/user/miniconda3/envs/mbe-phase2/bin/python`。**不要**用系统 python / base env。
- 任何新依赖：先 `pip install` 进 `mbe-phase2`，再同步写进 `requirements.txt`（版本 floor `>=x.y.z` 风格，与现有条目一致）。
- HF 模型下载走 `HF_ENDPOINT=https://hf-mirror.com`（已在 `.env`），ModelScope 镜像优先。

### 2. 三条标准命令

```bash
# 起服务（前端 + API + ingest worker 一体）
/home/user/miniconda3/envs/mbe-phase2/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 跑测试
/home/user/miniconda3/envs/mbe-phase2/bin/python -m pytest tests/ -x

# 跑评测（Phase E 上线后）
/home/user/miniconda3/envs/mbe-phase2/bin/python -m app.eval_harness --dataset audiovisual --n 20
```

### 3. 项目约束（不可越线）

- **NExT-GQA 已废弃**，所有评测路径指向自建音画 QA 数据集（Phase E）。不要再往 NExT-GQA 加代码。
- Eval 默认 `n=20`，不是 100（学生 token 预算）。
- **不引入 LLM-as-judge**。Phase E 评分 = 关键词命中 + citation 覆盖率，确定性的规则评分。
- 对话语言中文、代码 / 标识符 / 提交信息英文（提交信息允许中英混合，看现有 commit 风格）。
- 用户的 `.env` 真实 API key 不允许进 git；任何新 key 用占位符写到示例并 `.gitignore`。

### 4. 关键文件 map（Phase A–D 主要会动这些）

| 文件 | 角色 | 关键约定 |
| --- | --- | --- |
| [app/tools.py](app/tools.py) | LangGraph tool 集合 | `@tool` + `Annotated[..., InjectedState]` + 返回 `Command(update=..., goto=...)`；新工具需注册到模块底部 `TOOLS = [...]` |
| [app/cache.py](app/cache.py) | 一视频一目录约定 | `video_cache_dir(video_id)`、`load_meta/save_meta`；所有视频相关 artifact 都进 `data/cache/{video_id}/` |
| [app/preprocess.py](app/preprocess.py) | ingest pipeline | `preprocess_video()` 是入口；新阶段挂在这里，必要时与现有 stage 并行 |
| [app/progress.py](app/progress.py) | 进度推送 | `publish_progress(video_id, stage, label, ratio, error?)` + `stage_label` 字典；新阶段必须先在 `stage_label` 加 key |
| [app/graph.py](app/graph.py) | LangGraph 编译 + State | `GraphState` 用 `add_messages` reducer for messages、`_last_write` for 其余字段；新 state 字段必须配 reducer；`sanitize_dangling_tool_calls` 入口在 `chat_stream` 已挂 |
| [app/vqa.py](app/vqa.py) | VLM 客户端 + system prompt | `QA_SYSTEM_PROMPT` 决定 citation 协议、`answer_question` 是 VQA 唯一入口；改 prompt 必须 bump `AGENT_CODE_VERSION` |
| [app/eval_fingerprint.py](app/eval_fingerprint.py) | prediction-cache 版本 | `AGENT_CODE_VERSION` 凡是 prompt / 工具集 / 决策行为改了都要 bump；切 VLM/Orchestrator 模型也必须 bump |
| [app/main.py](app/main.py) | FastAPI 入口 | `/api/chat_stream` 是 SSE 流；新 event type 同步改 `app/static/app.js` |
| [app/models.py](app/models.py) | 模型加载 / 卸载 | `get_xxx / release_xxx` 模式做 VRAM 轮换；新模型必须按这个模式加，不要全局常驻 |
| [app/static/app.js](app/static/app.js) | 前端渲染 | `renderAnswer()` 已处理 markdown + KaTeX + `[FRAME:t=X]` 占位回填；新 citation kind 在这里扩展 |

### 5. Citation 协议（Phase B/C/D 反复改）

任何新增 citation kind 都要**同步改三处**，少一处都会出 bug：

1. `app/vqa.py` `QA_SYSTEM_PROMPT` — 告诉 VLM 怎么生成
2. `app/tools.py` `_grounding_report()`（约 L80）— 服务端校验
3. `app/static/app.js` `renderAnswer()` 里的 `marker` 正则 — 前端识别 + 占位回填

### 6. Phase 完成定义 (DoD)

每个 Phase 收尾前必须满足：

- [ ] 有 `tests/` 覆盖（单测优先；E2E 测试用 `data/uploads/reinforce_vs_A2C_remux.mp4` 作 fixture）
- [ ] `requirements.txt` 同步新依赖
- [ ] 如有 prompt / 工具集 / 决策行为变化 → bump `AGENT_CODE_VERSION`
- [ ] 新增 ingest 阶段在 `stage_label` + 进度条占合理比例
- [ ] 改了 tools 必须同步 grounding 校验
- [ ] commit message 中英文都行，要说明「为什么改」不是「改了什么」

### 7. 必须先问用户的事（不要擅自做）

- 切 LLM / VLM provider（当前 VLM=Bailian qwen3.6-plus，Orchestrator=DeepSeek deepseek-v4-flash）
- 改 checkpointer / DB schema（`data/graph_checkpoints.sqlite3` 已有 production 数据）
- `git push` 远端
- destructive git（`reset --hard`、`branch -D`、`push --force`）
- 改 `.env` 真实 key（仅可写占位符到 example）
- 跳过 `AGENT_CODE_VERSION` bump（prediction_cache 污染过一次了，痛过）

### 8. 当前 baseline（实施任何 Phase 前的起点）

- git HEAD = `03b6e6b`（"Heal dangling tool_calls in checkpoint before each turn"）
- `AGENT_CODE_VERSION = "v14"`
- VLM: Bailian qwen3.6-plus（DashScope chat_completions），native multimodal Early Fusion，1M context，reasoning model
- Orchestrator: DeepSeek `deepseek-v4-flash`（官方 API，已充值）
- LangMem: Doubao mini（火山引擎）
- Judge: Doubao lite（Phase E 不再用，留 backup）
- 本地 ASR (SenseVoice-Small + FSMN-VAD) 已 import-ready，未集成 ingest
- 前端 markdown + KaTeX 已上
- checkpoint guard 已上（`sanitize_dangling_tool_calls`）

### 9. 端到端冒烟视频

`data/uploads/reinforce_vs_A2C_remux.mp4`（王树森 RL 讲座，7'43"）+ Q1/Q2/Q3 三档（见下「视觉-only 基线」节）是固定回归用例。每个 Phase 完成后跑一遍：

- Phase A 完成：transcripts.jsonl 写出，肉眼检查 REINFORCE / A2C / baseline 等术语转写正确
- Phase B 完成：Q2 应命中 REINFORCE（被 transcript 召回）
- Phase C 完成：Q3 不加 "根据视频" 前缀也能调视频；答案有 `[TRANSCRIPT:t=...]` 引用，超过 PPT 一句话级别的对比
- Phase D 完成：能回答「第 N 张 PPT 写了什么」

### 10. 通用 hygiene

- 新 SQLite 库写到 `data/`，独立于 `mr_big_eye.sqlite3`（用户/会话域）和 `graph_checkpoints.sqlite3`（LangGraph 域）
- 新模型走 `app/models.py` 的 `get_*/release_*` 轮换模式；8GB VRAM 预算紧，常驻只允许 ASR + (CLIP or bge-m3)
- 新 user-facing 文案中文；日志 / 错误消息英文
- 任何 `print` 改 `logger.info/warning/error`，不要污染 stdout（uvicorn 共用）
- 时间戳全部秒为单位 float，不要 ms 和 s 混用

---

## 视觉-only 基线（Phase A 启动前测得，2026-05-26）

测试视频：`data/uploads/reinforce_vs_A2C_remux.mp4`（王树森 RL 讲座 *REINFORCE vs A2C*，7'43"，1280x720）。
VLM：qwen3.6-plus（百炼），Orchestrator：deepseek-v4-flash，AGENT_CODE_VERSION=v14。

| Q | 问题 | 答案要点 | Citation | 评价 |
| --- | --- | --- | --- | --- |
| Q1 (Easy / OCR) | 视频里的 PPT 上写了哪些公式或数学符号？ | 正确识别 transition tuple $(s_t,a_t,r_t,s_{t+1})$、单步 TD target $y_t=r_t+\gamma v(s_{t+1};w)$、多步 TD target $y_t=\sum_{l=0}^{m-1}\gamma^l r_{t+l}+\gamma^m v(s_{t+m};w)$ | 73/78/83/320s 全部精准指 PPT 帧 | ✅ 视觉强项，公式 LaTeX 正确无幻觉 |
| Q2 (Medium / multi-frame) | 视频中提到了哪几种强化学习算法或方法？ | 识别出 A2C、A2C with Multi-Step TD Target、One-step TD target；**漏掉 REINFORCE** | 71/76/80/62/67/150s 都是 A2C 类 PPT | ⚠️ PPT 主要展示 A2C，REINFORCE 几乎只在口述中出现 → 纯视觉天花板暴露 |
| Q3 (Hard / audio-dep) | 请详细对比 REINFORCE 和 A2C 这两个算法的区别。 | Orchestrator 判定"纯知识性问题，不调视频"，完全 bypass video，用 LLM 通用知识答题 | **零引用** | ❌ 音画联合刚需的反面教材——视觉不足以回答时不是降级，而是直接放弃视频，回答与视频本身脱钩 |

### 关键启示（直接影响 Phase A–C 设计）

1. **Q2 现象 → Phase A/B 必须把 ASR 接进检索栈**。视频里 REINFORCE 的关键内容（公式、与 A2C 的对比逻辑）只在讲师口述中，PPT 完全无痕。这正是 transcript 检索的核心价值证明，不再是"锦上添花"。
2. **Q3 现象 → Phase C router 不能简单做"视觉够不够"判断**。当前 orchestrator 在视觉证据不足时直接走"通用问答"路径，把视频丢了。Phase C 的 Question Router 必须包含一条规则：**只要问题里出现视频中的实体名/术语**（REINFORCE、A2C），就强制至少一次 transcript 检索，不允许 bypass。
   - **追加观察**：Q3 改成"**根据视频**，请详细对比……"后 orchestrator 就会调视频工具，答案落到 PPT *TD Target versus Return* 一张幻灯片（t=430s）上，只能给出"A2C 含 bootstrapping、REINFORCE 不含"这一句话级别的对比。说明 router 决策门槛是**问题里是否显式提到视频**，而非问题本身的语义；同时也再次印证 PPT 上能拿到的内容远比音频里讲到的少。Phase C router 应在分类阶段就默认 `video-grounded`，把"非视频问题"作为需要明确证据才能切走的少数情况。
3. **Q1 现象 → Phase D OCR 优先级可降**。qwen3.6-plus 在 PPT 公式上的 OCR + LaTeX 还原能力已经很强，纯 PPT 课堂场景下，PaddleOCR/pix2tex 的边际收益可能没原计划那么高。Phase D 之前先用真实评测（Phase E）量化一下 VLM OCR 召回率，再决定要不要做。
4. **前端缺 markdown 渲染**：答案里大量 `$...$` LaTeX、`### 标题`、`| 表格 |` 都是原文显示，体感差。Phase F 不能放到最后——至少 markdown + KaTeX 在 Phase A 完成前就该补上，否则后续测试都很痛苦。

### 已发现并修复的工程问题

- **decord EOF**：B 站 XCoder 重封装的 mp4 会让 decord 在 `_probe()` 抛 EOF。解决方案是 ffmpeg lossless remux（见下方 Phase A.0）。
- **main.py:376 `'list' object has no attribute 'get'`**：LangGraph 1.2 在 ToolNode 并发执行多个 tool 时，把 Command updates 以 list 形式 stream；老代码当 dict 处理就崩。已修。
- **Checkpoint 污染**：tool_node 异常中断时 AIMessage(tool_calls) 已落 checkpoint 但 ToolMessage 没跟上，下次 replay DeepSeek 直接 400。临时解法是手工新建 session；后面 Phase A 之后应加 guard——在 orchestrator 抛错时把当前 turn 的 messages 整体回滚。

---

## Phase A.0 · Ingest 兜底：ffmpeg remux preflight（半天，先做）

在做 Phase A 任何 ASR 集成前，先把 ingest 的「读不了视频」边缘修掉。

| 任务 | 做法 |
| --- | --- |
| 触发点 | [app/preprocess.py](app/preprocess.py) `_probe()` 用 decord 打开视频。增加 `try/except DECORDError`，捕获 EOF 类错误 |
| 兜底操作 | 用 imageio-ffmpeg 自带的 ffmpeg 跑 `-c copy -movflags +faststart` 重封装到 `data/cache/{video_id}/remuxed.mp4`，无损、~1s 完成（7 分钟视频 ≈ 0.5s） |
| 回灌 | 重封装后用新文件替换 `video_path` 重试 probe；成功就继续 ingest，失败再抛 |
| 元数据 | `meta.json` 加 `remuxed: true / source_remux_reason: "decord_eof"`，方便日后排查 |
| 触发场景 | B 站 XCoder、抖音、剪映等导出的 mp4 都可能 trip decord。已在 `reinforce vs A2C.mp4` 上验证有效 |
| 版本 bump | 不算 prediction-affecting，**不 bump** AGENT_CODE_VERSION |

**为什么放在 Phase A 之前**：ASR 也依赖 ffmpeg 抽 wav，但 ASR 的 ffmpeg 调用走 shell `subprocess.run`，对 B 站 mp4 是 OK 的（ffmpeg 本身能读，只是 decord 不行）。所以 ASR 不会因为这条坑被卡住，但 ingest 的 dense_frame_index / scene_caption 会。先把 decord 这条路打通，Phase A 才能安心并行接 ASR。

---

## Phase A · ASR 后处理与持久化（1–2 天）

把 `transcribe()` 的原始输出变成系统可用的资产。

| 任务 | 选型 / 做法 |
| --- | --- |
| Emoji / 表情符过滤 | SenseVoice 会把语气词输出成 emoji（😊😡），用正则剥离 |
| 术语归一化 | 简单 dict：「a 二 c」→「A2C」、「rl」→「RL」，覆盖课堂常见术语 |
| Hotwords 注入 | SenseVoice 支持 `hotword=` 参数，按视频上传时附带的关键词列表注入 |
| 持久化 | 每条视频写 `data/cache/{video_id}/transcripts.jsonl`（沿用现有「一视频一 cache_dir」约定，**不要**新开 `data/transcripts/`），结构 `{text, t_start, t_end, source: "asr"}` |
| Ingest 集成 | 在 [app/preprocess.py](app/preprocess.py) `preprocess_video()` 内插 ASR；并行点是「与 `caption_scenes`（远程 VLM）+ `dense_frame_index`（本地 CLIP）`asyncio.gather`」——ASR 用本地 GPU、caption 用远程 API、CLIP 抽帧偏 CPU，资源不冲突 |
| Progress stage | [app/progress.py](app/progress.py) `stage_label` 字典加 `"asr"`；ingest 总进度条预留 10–15% 给 ASR |
| Hotwords | SenseVoice `hotword=` 参数本 phase 只**留接口**（`transcribe(video_path, *, language, hotwords=None)`），默认 None；前端 textarea 在 Phase F 接入 |
| 元数据 | [app/cache.py:20](app/cache.py#L20) 定义的 `data/cache/{video_id}/meta.json` 增加 `has_transcript: true`、`asr_model`、`asr_duration_sec`、`transcript_segment_count` |
| 版本 bump | 完成后 `AGENT_CODE_VERSION` v13 → v14（transcript 进入 state，缓存预测应失效） |

**产出**：每个视频上传后自动有可索引的 transcript。

## Phase B · 文本检索栈（3–5 天）

让 transcript 可被 orchestrator 召回。

| 任务 | 选型 / 做法 |
| --- | --- |
| 分块策略 | 滑窗 3–5 句一块，1 句重叠；每块带 `[t_start, t_end]` |
| 稀疏检索 | SQLite FTS5 + jieba 分词，中文短查询友好 |
| FTS5 库位置 | **新建** `data/transcripts.sqlite3`，不混入 `data/mr_big_eye.sqlite3`（后者是用户/会话域）。解耦更干净，迁移/清理也好做 |
| 稠密检索 | **bge-m3**（多语种，2.3GB fp16，中英 code-switch 友好），ChromaDB 复用现有基建 |
| 模型管理 | [app/models.py](app/models.py) 现有 `get_bge / release_bge`（指 CLIP/BGE-vision）。新增独立 `get_text_embed / release_text_embed`，不混用。8GB VRAM 下采用「检索阶段 release CLIP / caption 阶段 release text_embed」轮换；ASR 模型常驻 |
| 融合 | RRF（Reciprocal Rank Fusion），k=60，top-10 → top-5 截断 |
| 新工具 | `retrieve_transcript_evidence(query, top_k)` + `search_transcript_keyword(keyword)`（精确词命中场景） |
| 引用格式 | `[TRANSCRIPT:t=10.5-15.2]`，与 `[FRAME:t=...]` 并存 |
| 版本 bump | 完成后 v14 → v15 |

**产出**：orchestrator 能用 transcript 回答「老师提到 baseline 的好处是什么？」这种纯听觉问题。

## Phase C · 音画联合（约 1 周，里程碑）

核心阶段——让两种模态真正协同。

| 任务 | 选型 / 做法 |
| --- | --- |
| Question Router | 轻量分类器或基于关键词的启发式，输出 `audio_heavy / visual_heavy / joint / general` |
| 联合对齐工具 | `align_audiovisual_evidence(timestamp, window_sec)`：在同一时间窗内同时返回 frames + transcript 片段 |
| VLM 提示词扩展 | 在系统提示里说明 `[TRANSCRIPT:t=...]` 引用规则；允许引用 transcript 而不一定要 frame |
| Citation 校验 | 改 [app/tools.py](app/tools.py) `_grounding_report()`（line 80 附近）：transcript-only 答案接受 `[TRANSCRIPT:t=...]`；audio-only / joint 模态分别有最小引用要求 |
| 前端 citation 渲染 | [app/static/app.js](app/static/app.js) 的 `[FRAME:t=...]` 正则要扩展到 `[TRANSCRIPT:t=...]`，点击跳转到对应时间 |
| Orchestrator 路由 | router 输出影响首调用工具的优先级：joint → align；audio_heavy → retrieve_transcript_evidence |
| 版本 bump | 完成后 v15 → v16（提示词 + 工具集合都变了） |

**产出**：典型问题如「老师讲解 A2C 时屏幕上对应的公式是什么？」可以工作。

## Phase D · 课堂垂直增强（3–5 天）

针对 PPT/板书录像，做模态特化。

| 任务 | 选型 / 做法 |
| --- | --- |
| 关键帧检测 | imagehash dHash + cv2 帧差，AND-fusion 触发 slide change；显著减少冗余帧 |
| 板书/PPT OCR | **PaddleOCR PP-OCRv4** 中文专用；图像区域 ROI 用现有 CLIP scene |
| 公式识别（可选） | pix2tex 或 Nougat，仅在检测到公式区域时调用 |
| Slide 索引 | OCR 文本作为第三个检索通道（slide_chunks），同样进 FTS5 + bge-m3 |
| 引用格式 | `[SLIDE:t=...]`；前端 app.js 正则、`_grounding_report` 都要同步扩展 |
| 版本 bump | 完成后 v16 → v17 |

**产出**：能回答「第 3 张 PPT 里的公式是什么？」「老师板书写的最后一个词」。

## Phase E · 评测重建（3 天）

放弃 NExT-GQA，搭建真实评测。

| 任务 | 做法 |
| --- | --- |
| 数据集 | 20–30 条视频（课堂、讲座、教程），每条 5–10 分钟 |
| 题目 | 手工标注 150–200 题，覆盖 4 类： audio 30% / visual 20% / joint 35% / overview 15% |
| 题目 schema | 单题 JSON：`{question_id, video_id, question, modality_tag: "audio"\|"visual"\|"joint"\|"overview", question_type, expected_keywords: [str], expected_citation_min: int, expected_citation_kinds: ["frame"\|"transcript"\|"slide"], reference_answer?: str}`；存 `eval/audiovisual/questions.jsonl` |
| 评分 | 关键词命中 + citation 覆盖率，**不用 LLM-as-judge**（不可控、不稳定） |
| 形式 | 开放问答 + MCQ 混合；MCQ 用现有硬规则评分 |
| CI | `python -m app.eval_harness --dataset audiovisual --n 30` |

**产出**：一个可信、可复现的进步衡量工具，每次改动都可以跑。

## Phase F · 前端与产品化（2–4 天）

把能力包装成日常可用的工具。

| 任务 | 做法 |
| --- | --- |
| Transcript 时间轴侧栏 | 显示完整 transcript，点击跳转到视频对应时刻 |
| 可点击的 citation | 答案里的 `[FRAME:t=4.0]` `[TRANSCRIPT:t=10.5-15.2]` 都能点击跳转 |
| Ingest 进度条 | ASR / CLIP / scene 三个子步骤分别显示进度 |
| 字幕导出 | VTT 格式，可下载或外挂播放器 |
| 上传时 hotwords | 一个简单 textarea，让用户填课程关键词（人名、术语） |

## Phase G · 性能与边缘（持续）

不在 MVP 路径上但终究要面对的。

- 长视频内存：>15 分钟视频的 transcript chunk 数 → 检索 latency
- 中英 code-switch：SenseVoice + bge-m3 都支持，但需要实测
- 噪声鲁棒：VAD 阈值、低质音频
- 检索延迟：dense + sparse + RRF 的端到端 latency 控制在 500ms 内

---

## 技术栈总表

| 模块 | 选型 | VRAM / 磁盘 |
| --- | --- | --- |
| ASR | SenseVoice-Small + FSMN-VAD（已就位） | ~1GB VRAM, 250MB |
| Hotwords | SenseVoice 内置 | — |
| 文本稀疏检索 | SQLite FTS5 + jieba | 磁盘 |
| 文本稠密 embedding | bge-m3 fp16 | ~2.3GB VRAM |
| 检索融合 | RRF (k=60) | — |
| 关键帧检测 | imagehash dHash + cv2 | CPU |
| PPT OCR | PaddleOCR PP-OCRv4 中文 | ~500MB |
| 公式识别 | pix2tex（可选） | ~400MB |
| 评测 | 自建 150 题音画 QA + 关键词/citation 评分 | — |

VRAM 预算总览（RTX 4060 8GB）：CLIP (~200MB) + ASR (~1GB) + bge-m3 (~2.3GB) + VLM client（远程）+ PaddleOCR (~500MB) ≈ 4GB peak，留 4GB 给系统/burst。

---

## 顺序建议

> A → B → C → E.1–E.2 → F → D → G

**理由**：
- A/B/C 是新增能力的关键路径，先打地基。
- E 提前到 D 前面，是为了在投入 OCR/公式识别前用真实评测验证音画联合是否真的解决了用户问题——避免 D 做了一半发现瓶颈在别处。
- F 早一点上，是因为产品体感反馈比 benchmark 更能指明下一步该做什么。
- D 在 E/F 之后，是为了让 OCR/公式识别的优先级由实际用户使用决定，而不是凭直觉。
- G 贯穿全程。

**总工时估算**：3–4 周高强度，或 6–8 周课余/兼职节奏。

---

## 现在能立刻动的事（Phase A 启动）

1. 在 `app/asr.py` 顶部加 `_EMOJI_RE` + `_TERM_DICT`，在 `transcribe()` 输出前做一遍后处理；`transcribe()` 签名加 `hotwords: list[str] | None = None`（本 phase 默认 None）。
2. 在 [app/preprocess.py](app/preprocess.py) `preprocess_video()` 中把 ASR 作为新阶段，和 `caption_scenes` + `dense_frame_index` 用 `asyncio.gather` 并行。
3. 写入 `data/cache/{video_id}/transcripts.jsonl`（**不要**新建 `data/transcripts/`）。
4. `data/cache/{video_id}/meta.json` 增加 `has_transcript`、`asr_model`、`asr_duration_sec`、`transcript_segment_count`。
5. [app/progress.py](app/progress.py) `stage_label` 加 `"asr"`。
6. 在 `data/uploads/reinforce vs A2C.mp4` 上跑 end-to-end ingest，肉眼检查 transcript 质量。
7. 把 `funasr`、`soundfile`、`imageio-ffmpeg`（已加）这一段在新机器上 `pip install -r requirements.txt` 验证一次。
8. 完成后 bump [app/eval_fingerprint.py](app/eval_fingerprint.py) `AGENT_CODE_VERSION` v13 → v14。

**环境提醒**：所有安装都进 `mbe-phase2` conda env（`/home/user/miniconda3/envs/mbe-phase2/bin/python`），新组件同步更新 `requirements.txt`。

**版本管理通则**：每个 Phase 完成后 bump 一次 `AGENT_CODE_VERSION`（A→v14, B→v15, C→v16, D→v17）。模型/提示词/工具集合任一变化都要 bump，否则 prediction_cache 会污染。
