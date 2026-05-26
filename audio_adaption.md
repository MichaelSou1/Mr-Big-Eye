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
