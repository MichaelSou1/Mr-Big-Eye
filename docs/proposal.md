研究探索复述

背景

你是算力受限(4×RTX3080-20GB,PCIe 无 NVLink)的研究者,想做小参数视觉模型(≤7B)方向。起点是你已有的 agent 项目 Mr-Big-Eye——一个长视频问答系统:多模型协作(独立 orchestrator + 远程 VLM API + 检索器),14 个工具,LangGraph 编排,推理时多步调用工具来定位证据、回答问题。

驱动这个研究的核心直觉:当前的 harness(脚手架)会随着模型能力增强而被”吞掉”。 我们把这个直觉拆精确了——harness 有三类:提供外部能力的(物理性,吞不掉)、提供确定性保证的(认识论性,吞不掉)、弥补当前模型缺陷的临时补丁(会被吞)。你要做的不是被动等模型迭代吞掉补丁,而是主动把第三类临时脚手架内化进模型权重。

中途的关键发现:最接近的工作 VideoTemp-o3(快手,2026.3)和你撞在同一研究家族,但走了相反的路。它把多模型协作合并成单个 VLM 的多轮自调度——但推理时工具循环依然在场(localize-crop-answer 还在跑,外部裁剪模块还在)。你的判断很准:它把 orchestrator 融进了 VLM。但需要校准一格:它简化的是”模型数量”,没简化”推理时步数”。

目的

核心假设:小 VLM(Qwen3-VL-4B-Instruct)能把多步 agentic 视觉推理内化成单次前向的 CoT,使其微调后在一次前向、无任何工具的情况下推理更强;并且能用因果探针证明这个增益是泛化推理,而非模板记忆。

和 VideoTemp-o3 的精确分界(按”推理时工具是否在场”划,不按模型数量划):它内化了”何时何地裁剪”的策略但推理时仍调工具;你要再深入一层,把整个多轮自调度从运行时挪进训练时——推理时不输出 tool_call、无外部裁剪、纯单次前向只靠看得到的帧作答。

卖点是方法和”可内化性边界”(哪些推理类型能内化、哪些只能记忆),不是榜单分数。 两个不可让渡的卖点:推理时无工具/单次前向,以及因果验证内化的是真推理。绝不拿 “single model” 当卖点(VideoTemp-o3 占了)。

两个硬约束

	1.	不依赖更强 teacher(self-improvement framing):生成训练轨迹的 agent 必须用 4B 自己当 VLM backbone,不能用远程大模型。唯一例外是 rewriter(只改文本不注入视觉能力)可用大模型。这也意味着你旧的 distillation 轨迹(teacher 是 Qwen3-VL API)不能复用——不是数据脏,是叙事脏。
	2.	只内化 type-1 步骤:轨迹里的步骤分两类。type-1(可内化)是单次 VLM 本就能做的纯视觉推理(数物体、比颜色、读画面里的钟、排事件顺序),能锚到 [FRAME:t=];type-2(必须剔除)是依赖外部工具的步骤(检索排序、temporal resolver、grounding 校验、跨帧匹配分)。把 type-2 塞进 CoT 就是教模型幻觉。

范围已定死 vision-only:不做 audio/transcript/OCR 文本,既让因果归因干净,又大幅提高留存率。答案本质依赖音画的整道题直接丢弃。

完整流程

Phase 0 — 接本地 4B backbone(唯一要新写的基础设施)
在 vqa.py 加 VLMBackbone 抽象层,让 agent 的 VLM 可切换远程/本地。你的整套 indexing、agent loop、eval harness 全部复用,不重写。

Phase 1 — 轨迹生成
4B 当 backbone,在 NExT-GQA 视频上跑现有 agent loop,落全量结构化轨迹。

Phase 2 — 严格过滤
复用 harness 但关掉 Soft-Waive(它”答案对就过”,会放进蒙对但过程错的轨迹,污染训练)。要求:答案对 + failure_tags 空 + grounding 命中(IoP≥0.5)。再加 vision-only 的 case 级范围过滤。

Phase 3 — 轨迹改写成 CoT(新模块)
rewriter 把轨迹改写成第一人称单次视觉推理链:只保留 type-1、工具结论转成”画面里可见”的前提、删掉所有工具名/检索动词/非视觉数字。代码后置校验:只允许 [FRAME:t=] 标记。

Phase 4 — 一致性过滤(决定性的一步)
采用 VideoTemp-o3 两步验证设计(充分性 + 一致性),但 teacher 换成 4B 自己。把改写后的 CoT 喂回 4B,只给帧不给工具,能复现正确答案才留——保证每条 CoT 都是 4B”够得着”的。

关键 gate:先在 50-case 上跑 Phase 0→4,看留存率。<30% 停(任务对 4B 太难),≥50% 才放大。1-2 天出结果,决定项目可行性。

Phase 5 — SFT(LoRA)
base 钉死 Qwen3-VL-4B-Instruct(不是 Thinking——Instruct 几乎无自发 CoT,因果归因才干净)。1k–3k 条干净 CoT,不超 5k(防过拟合毁 OOD 实验)。SFT 前先 probe base 会不会自发 CoT。

Phase 6 — GRPO/DAPO
SFT 是行为克隆有天花板,RL 才能超越并纠错。reward:答案正确 + grounding marker 有效 + penalty-aware IoU(σ=0.1, λ=0.1,把 CoT 里声称看的时刻当隐式时序 claim 对照 GT)。O-LoRA 防遗忘。

Phase 7 — 因果探针(科学核心)
区分”真推理”vs”模板记忆”:

	•	实验1 分布偏移(L1表面/L2组合/L3深度)+ shuffled-CoT 对照组。
	•	实验2 CoT 因果干预(篡改中间结论看答案翻转率 / CoT 消融 / 早停探针)——最决定性。
	•	实验3 视觉依赖探针(图像扰动 / 反事实图像 / 注意力检验)。
	•	实验4 数据诊断曲线。
	•	样本量:20-case 只做方向 sanity,正式统计先跑 power_analysis 反推 N(约 300–500,video-isolated)。

Phase 9(可选)— 内化 vs 原生机制对比
对你内化的 Instruct 模型和原生 Thinking-4B 做同样的因果干预,比翻转率——这本身是论文级发现。

数据与隔离

主力训练源 NExT-GQA(唯一同时满足纯视觉+grounding GT+可验证);Video-MME 纯视觉子集作补充;WorldSense 弃用(强制音画)。下载 ModelScope 优先,但 NExT-GQA 大概率无镜像,标注走原 repo,视频找 NExT-QA 镜像。训练用 NExT-GQA 全量,评测用无 video 重叠的集(天然构成 L1 OOD)。VideoTemp-Bench 只用 >3min 档(0–3min 档是从 NExT-GQA 抽的,会泄漏),仅作 OOD 评测。

---

Implementation deltas / 设计修订(2026-06-08)

一次设计审查发现:原方案会让 50-case 闸门“因为错误的原因而通过”,核心症结是「推理时帧从哪来」没定义。据此把命题收窄,并落地了 7 处改动。原文以上保留不动,以本节为准。

命题收窄(诚实版):从“内化整个 agent 的多步推理”收窄为——**在“相关帧可见”的短视频域,把工具脚手架式的多步视觉推理压进一次无工具前向,并用因果探针证明是真推理、刻画可内化性边界。** 两个不可让渡卖点不变(单次前向/无迭代工具、因果验证);**“选哪帧”这个能力显式声明为不可内化、范围外**——这把 type-1/type-2 边界从“工具名”下沉到“能力”层面。NExT-GQA 视频短(~40s @1fps),固定均匀采样大概率已含证据,正是这个域里选帧 confound 很小的原因。

1. **固定均匀采样器(新增,linchpin)。** 推理/训练/CoT/一致性过滤统一用一个 query-agnostic、训练-free 的 16 帧均匀采样器(一次性预处理函数,非 per-query 工具循环,所以“无工具”诚实成立)。它对 base 与内化模型一致,从而把“选帧”这一变量钉死,只测推理。代码:`app/distill/frames.py`、配置 `distill_sampler_frames=16`。
2. **Phase 0 流程图新增“无工具选帧”一步。** 帧来源不再是 agent 检索结果,而是上述采样器;SFT 用的也是这套帧,消除 train/inference 不一致。
3. **Phase 2 加“证据覆盖”过滤。** 均匀采样若没覆盖到 GT 证据窗,则该 case 出范围丢弃(`evidence_not_in_uniform_sample`);需要检索才找得到针的 case = 需要不可内化的选帧能力,本就该丢。无 GT 的补充集跳过此检查并打标。
4. **Phase 4 闸门改成测“可达性”而非“抄答案”。** seed CoT 先剥掉结尾答案;保留判据改为 **conditioned 对 ∧ stable ∧ free-form 错**——CoT 必须真帮上忙,且 base 单次前向本来答不对才算有内化信号。闸门头号指标改为 `signal_gain_rate`(会比旧 retention 低,这是诚实化后的预期)。
5. **数据隔离改为 train/held-out 切分。** 不再 100% 拿 NExT-GQA 训练;按 video 切一个 disjoint held-out 作为 IID 评测点,§7.1 的 IID→OOD 斜率才有定义。代码:`scripts/split_cases.py`。
6. **Phase 6 暂关 temporal-IoU 奖励。** 固定帧无工具下,CoT 引用的 [FRAME:t=] 由采样器决定而非模型动作,IoU 奖励不 load-bearing 且可 hack,故只留答案正确 + marker 有效;GT 仍透传供 7.3c 探针用。
7. **7.3c 探针更名。** OpenAI 兼容端点取不到 attention,故由“attention check”改为“声称时刻 vs GT 一致性”检查(CoT 引用的 [FRAME:t=] 是否落在 grounding GT 上)。

附带修复:grounding GT(gold_timestamps/gold_scenes)现已从轨迹透传到 CoT/SFT/RL 产物(原先在 CoT 阶段被丢)。

可行性结论:即使收窄后“可内化信号集”很小,也是落在“可内化性边界”这一既定卖点里的可发表边界结果——故项目仍值得跑;50-case 闸门 1–2 天给出诚实判断。

