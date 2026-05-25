# LensWalk-inspired agent upgrade — implementation plan

**Source paper:** `paper_references/LensWalk.pdf` (Li et al., 2026). Appendix A
(Figures 5-10, Algorithm 1, Section B.1 budgets, Section C.3 failure modes) is
the load-bearing reference. Re-read it before editing prompts.

**Status snapshot:** NExT-GQA 20-case sample @ seed 0 currently scores 14/20
(70%) with `Qwen3-VL-30B-A3B-Instruct` (VLM, ModelScope) + `doubao-seed-2-0-pro`
(orchestrator, Volcengine ARK). Bottleneck diagnosis: 4 both-wrong cases have
retrieval recall ≥ 0.5 — the VLM misreads visual content. Goal of this plan is
to give the agent richer multi-segment evidence shaping + cross-turn subject
memory so the VLM has a better chance.

**Expected outcome:** 14/20 → ~16-17/20 on NExT-GQA; better multi-turn product
UX via persistent subject registry. ~1.5 day total effort.

---

## Environment & constraints (do NOT violate)

- Conda env: `/home/user/miniconda3/envs/mbe-phase2/bin/python` (NEVER use base).
- Dataset scope: **NExT-GQA only**. LongVideoBench is descoped.
- Eval sample: 20 cases, `--sample-seed 0` (token budget).
- No proxy; ModelScope/Volcengine domestic CDN only.
- Disk-conscious — do not introduce new model downloads.
- Judge API: leave the `JUDGE_*` keys in `.env` for the user to fill in; never
  hardcode credentials.
- Per-call frame budgets must stay well under LensWalk's defaults (we have ~1/3
  the token budget): cap segment_focus at 12 frames, stitched_verify at 24
  frames, max_tool_calls stays at 8.
- Observer prompts → **Chinese**, code/identifiers stay English.

## Default eval command (after each phase)

```bash
/home/user/miniconda3/envs/mbe-phase2/bin/python scripts/eval_harness.py \
    --cases tests/fixtures/eval_cases_nextgqa.jsonl \
    --sample 20 --sample-seed 0 --judge \
    --prediction-cache data/eval/prediction_cache.jsonl \
    --output data/eval/runs/nextgqa_$(date +%Y%m%d_%H%M)_lensX.json
```

Each phase that changes prompts or agent runtime MUST bump
`AGENT_CODE_VERSION` in `app/eval_fingerprint.py` so the prediction cache
invalidates affected entries automatically. Bump v5 → v6 on first commit of
this plan, then once more after #5 lands if needed.

---

## #1 — `stitched_verify` tool (~4h, highest ROI)

Synthesize evidence across **multiple time windows** in a single VLM call. Used
when the question requires comparing or relating moments separated in time
(temporal_order, comparison, counting on dispersed events).

### Add tool in `app/tools.py`

```python
@tool
async def stitched_verify(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    question: Annotated[str, Field(description="原始用户问题。")],
    windows: Annotated[
        list[dict[str, float]],
        Field(description="待对比的时间窗列表，例如 [{\"start\":10.0,\"end\":15.0}, {\"start\":30.0,\"end\":34.0}]。最多 4 段。"),
    ],
    fps_per_window: Annotated[
        float, Field(description="每段抽帧 fps，默认 1.0；总帧数自动 cap 在 24。")
    ] = 1.0,
) -> Command:
    """Compare/synthesize evidence across 2-4 disjoint time windows."""
```

Implementation notes:
- Reuse `_load_dense_payloads` helper but pass per-window targets and dedup
  frames so total ≤ 24. If user asks for >4 windows, truncate to 4 + warn in
  payload (`{"warning": "truncated to 4 windows"}`).
- Sort windows by start time; sort frames by timestamp before passing to VLM
  so the model sees chronological order.
- Build the VLM call via `answer_question(question, frames, timestamps,
  history=None, system_prompt=STITCHED_VERIFY_PROMPT)` — see #2 for the
  refactor that exposes `system_prompt`.
- Output payload schema:
  ```python
  {
    "tool": "stitched_verify",
    "windows": [{"start":..., "end":..., "frame_count":...}, ...],
    "answer": "<vlm response>",
    "subject_deltas": [...],   # see #5
    "next": "verify_grounding",
  }
  ```

### Observer prompt (Chinese, store as `STITCHED_VERIFY_PROMPT` in `app/vqa.py`)

```
你是视频分析专家，专门从多个独立的关键时刻中综合信息。
你看到的是从同一视频不同时间段抽取的若干帧，每帧带 [t=Xs] 时间戳。
你的任务：
1. 简要描述每个时间段呈现的内容（按时间顺序）。
2. 指出不同时刻之间的联系、变化或对比。
3. 严格基于所给帧回答；不要推测帧之间时间间隔内发生了什么。
4. 不要做超出帧内容的推断；如果证据不足以回答问题，明确说出来。
5. 所有可视事实必须给出 [FRAME:t=X.X] 引用；典型 2-4 个。
```

### Register in `TOOLS` list at the bottom of `app/tools.py` and add to the
orchestrator prompt's tool catalog.

### Orchestrator prompt addition (`app/graph.py:_orchestrator_prompt`, video branch)

Append to the existing guidance:
> 当问题需要比较不同时间段的事件（如顺序、对比、跨段计数），优先调用
> `stitched_verify` 而不是多次 `retrieve_video_evidence`。最多 4 段窗口、
> 总帧数自动限制在 24。

### Tests

- `tests/test_tools_planner.py`: assert `stitched_verify` truncates >4
  windows, dedupes overlapping frames, respects the 24-frame cap.
- `tests/test_graph_orchestrator.py`: smoke test that the model can call
  `stitched_verify` and the result merges into `retrieved_frames`.

---

## #2 — `segment_focus` tool (~2h)

Densely sample a **single** time window for fine visual detail. Replaces / wraps
the existing `expand_temporal_evidence` semantics with a LensWalk-style
Observer.

### Refactor `app/vqa.py:answer_question`

Add an optional `system_prompt` parameter that defaults to `QA_SYSTEM_PROMPT`
so each Observer tool (segment_focus, stitched_verify, answer_with_evidence)
can inject its own constrained prompt without forking the VLM call code.

```python
async def answer_question(
    question: str,
    frames: list[Image.Image],
    timestamps: list[float],
    history: list[dict[str, str]] | None,
    *,
    system_prompt: str | None = None,
    subject_registry: list[dict[str, Any]] | None = None,  # see #5
) -> str:
    ...
```

When `subject_registry` is non-empty, render it as a header injected
above the user question (see #5 for the template).

### Add tool in `app/tools.py`

```python
@tool
async def segment_focus(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    question: Annotated[str, Field(description="原始用户问题。")],
    center_t: Annotated[float, Field(description="窗口中心时间（秒）。")],
    half_window_sec: Annotated[float, Field(description="窗口半宽，默认 4 秒。")] = 4.0,
    fps: Annotated[float, Field(description="抽帧 fps，默认 1.0；最多 12 帧。")] = 1.0,
) -> Command:
    """Densely sample one short window for fine visual detail (≤12 frames)."""
```

Keep `expand_temporal_evidence` for backward compat but **deprecate** in the
orchestrator prompt — direct the model to call `segment_focus` instead.
Mark `expand_temporal_evidence` description as "(legacy; prefer segment_focus)".

### Observer prompt (Chinese, store as `SEGMENT_FOCUS_PROMPT` in `app/vqa.py`)

```
你是细致严谨的视频分析助手。
你看到的是同一片段内按时间顺序均匀抽取的若干帧，每帧带 [t=Xs] 时间戳。
你的任务：
1. 聚焦用户问题所关心的视觉细节，按帧描述变化。
2. 严格基于所给帧；不要推断该窗口之外发生的事，也不要想象帧之间的内容。
3. 如果细节模糊或证据不足，明确说"该窗口内看不清/看不到"。
4. 所有可视事实必须给出 [FRAME:t=X.X] 引用；典型 2-4 个。
```

### Orchestrator prompt addition (video branch)

> 当问题聚焦某一短窗口内的细节（动作识别、计数、文字识别），调用
> `segment_focus` 在该窗口密采 ≤12 帧；优先于 `expand_temporal_evidence`。

### Tests

- `tests/test_tools_planner.py`: segment_focus respects 12-frame cap, handles
  window clamping at video boundaries.
- `tests/test_vqa.py`: assert `answer_question` accepts and uses custom
  `system_prompt`.

---

## #3 — VQA prompt LensWalk-style scope constraint (~30 min)

Tighten `QA_SYSTEM_PROMPT` in `app/vqa.py` with the LensWalk Observer
discipline. Replace existing language about evidence with:

```
- 严格基于所提供的帧回答；不要推测帧之外或帧之间发生的事。
- 如果证据不足以判断，明确说"证据不足"，不要凑答案。
- 每个具体视觉论断都要给出 [FRAME:t=X.X] 引用；典型 2-4 个。
- 多选题（A/B/C/D/E）必须从给出的选项中选一个，除非证据明确矛盾。
```

Keep the existing positive guidance (warmth, conciseness, Chinese matching).

### Tests

`tests/test_vqa.py`: existing FRAME marker tests should still pass. Add an
assertion that the prompt mentions "证据不足" so future edits don't drift.

---

## #4 — Orchestrator THINK→PLAN→OBSERVE format (~30 min)

Inject LensWalk's Reasoner discipline into `app/graph.py:_orchestrator_prompt`
(video branch only). Add this paragraph:

> 每次调用工具之前，先用一句中文写出 PLAN：要解决什么子问题（goal）、用哪个
> 工具（tool）、看哪段时间（time_range）、抽帧策略（sampling）。
> 工具返回后，写一句 OBSERVE：这次拿到了什么、还差什么。
> 不要重复同样参数的工具调用。verify_grounding 返回 grounded=true 时立即
> 输出 draft 作为最终回答。

This piggybacks on `_dedup_tool_calls` and `_verify_grounding_stalled` already
in place. The behavioral change should be reasoning-trace clarity for doubao,
not new control flow.

### Tests

No new tests needed; existing `tests/test_graph_orchestrator.py` covers
loop-termination. Manually inspect 2-3 reasoning traces after the change to
confirm the model is actually writing PLAN/OBSERVE lines (in tool_calls' AI
message content).

---

## #5 — Subject Registry (~3h)

Intra-video, cross-turn entity tracking. **NOT** stored in LangMem (that's
cross-session, would pollute other videos). Stored in `GraphState` via
AsyncSqliteSaver checkpointer, scoped per `thread_id`.

### `GraphState` additions in `app/graph.py`

```python
class GraphState(TypedDict):
    ...
    subject_registry: Annotated[list[dict[str, Any]], _last_write]
```

Each subject:
```python
{
    "id": "person_A",
    "label": "红衣男子",
    "first_seen_t": 12.3,
    "last_seen_t": 45.6,
    "attributes": ["持有背包", "在跑步"],
    "evidence_frames": [12.3, 30.1],
}
```

### Subject delta protocol

Observer tools (`segment_focus`, `stitched_verify`, `answer_with_evidence`)
ask the VLM to emit `subject_deltas` as part of the response. To avoid a
second LLM call, embed the schema in the Observer prompt:

```
在答案末尾追加一行 JSON（不要任何 markdown 围栏），格式：
SUBJECT_DELTAS: {"deltas": [
  {"op": "add", "id": "person_A", "label": "红衣男子", "first_seen_t": 12.3,
   "attributes": ["持有背包"], "evidence_frames": [12.3]},
  {"op": "update", "id": "person_B", "attributes_add": ["在跑步"],
   "last_seen_t": 45.0, "evidence_frames_add": [45.0]}
]}
若无主体变化，输出 SUBJECT_DELTAS: {"deltas": []}。
```

### Python merge logic (new helper in `app/tools.py` or `app/memory.py`)

```python
SUBJECT_REGISTRY_MAX = 15

def merge_subject_deltas(
    registry: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {s["id"]: dict(s) for s in registry}
    for delta in deltas:
        op = delta.get("op")
        sid = delta.get("id")
        if not sid:
            continue
        if op == "add" and sid not in by_id:
            by_id[sid] = {
                "id": sid,
                "label": delta.get("label", sid),
                "first_seen_t": delta.get("first_seen_t"),
                "last_seen_t": delta.get("first_seen_t"),
                "attributes": list(delta.get("attributes") or []),
                "evidence_frames": list(delta.get("evidence_frames") or []),
            }
        elif op == "update" and sid in by_id:
            entry = by_id[sid]
            entry["attributes"] = sorted(set(entry["attributes"]) | set(delta.get("attributes_add") or []))
            entry["evidence_frames"] = sorted(set(entry["evidence_frames"]) | set(delta.get("evidence_frames_add") or []))
            new_last = delta.get("last_seen_t")
            if new_last is not None:
                entry["last_seen_t"] = max(entry.get("last_seen_t") or 0.0, float(new_last))
    merged = list(by_id.values())
    # Prune to max 15 by oldest last_seen_t
    merged.sort(key=lambda s: s.get("last_seen_t") or 0.0, reverse=True)
    return merged[:SUBJECT_REGISTRY_MAX]


def parse_subject_deltas(answer_text: str) -> tuple[str, list[dict[str, Any]]]:
    """Strip the trailing SUBJECT_DELTAS line from answer; return (clean_answer, deltas)."""
    # Robust parsing: look for last line starting with "SUBJECT_DELTAS:".
    # Return ("", []) on parse failure but keep the full answer.
```

### Wire into Observer tools

Each Observer tool:
1. Reads `state.get("subject_registry", [])` and passes it to `answer_question`.
2. Parses `subject_deltas` from the VLM response.
3. Returns `Command(..., update={"retrieved_frames": ..., "subject_registry":
   merge_subject_deltas(registry, deltas)})`.
4. Strips the SUBJECT_DELTAS line from the user-visible `answer` field of the
   tool payload (don't leak it to the next orchestrator round).

### Registry injection format (in `answer_question` when registry is non-empty)

Prepend to the user message (NOT the system prompt — keeps it visible to the
VLM without bloating cached system prompt tokens):

```
【已知主体登记表】（来自此前观察，可直接引用其 id/label 进行消歧）
- person_A (红衣男子, 首次见于 12.3s, 最近 45.6s): 持有背包, 在跑步
- person_B (...

【用户问题】
<question>
```

If registry is empty, skip the header entirely.

### Memory write node (`app/graph.py:_make_memory_write_node`)

Leave LangMem write path unchanged. Subject Registry lives in GraphState only,
so checkpointer persistence is automatic. **Do NOT** also push subjects to
LangMem — they're intra-video and would pollute cross-session memory.

### Tests

- New `tests/test_subject_registry.py`:
  - `merge_subject_deltas` add → present in result.
  - `merge_subject_deltas` update → attributes/evidence_frames union, last_seen_t = max.
  - Prune to 15 by `last_seen_t` desc.
  - `parse_subject_deltas` recovers deltas from realistic VLM output; returns
    `("", [])` on garbage trailing line (but keeps answer intact).
- `tests/test_graph_orchestrator.py`: smoke test that `subject_registry`
  survives a tool call round-trip via the checkpointer.

### Cache invalidation

This phase changes the VQA user-message structure when registry is non-empty.
Bump `AGENT_CODE_VERSION = "v7"` (assuming #1-#4 landed at v6).

---

## Verification gates (run after each phase)

```bash
# Tests
/home/user/miniconda3/envs/mbe-phase2/bin/python -m pytest tests/ -q

# Eval (regenerates predictions due to fingerprint bump)
/home/user/miniconda3/envs/mbe-phase2/bin/python scripts/eval_harness.py \
    --cases tests/fixtures/eval_cases_nextgqa.jsonl \
    --sample 20 --sample-seed 0 --judge \
    --prediction-cache data/eval/prediction_cache.jsonl \
    --output data/eval/runs/nextgqa_$(date +%Y%m%d_%H%M)_lens${PHASE}.json

# Sanity check the registry persists across turns (manual)
/home/user/miniconda3/envs/mbe-phase2/bin/python scripts/smoke_test.py
```

Pass criteria per phase:
- #1 stitched_verify: ≥1 of the 4 both-wrong cases (`5521781780-3`,
  `4798585428-2`, `11587211476-7`, `10109097475-2`) flips to PASS.
- #2 segment_focus: ≥1 additional flip among visual-detail questions.
- #3, #4: no regressions; reasoning traces show PLAN/OBSERVE lines.
- #5: smoke test shows registry populated after turn 1, referenced in
  turn 2 VQA prompt.

---

## Implementation order (recommended)

1. **#1 + #5 first** — Subject Registry delta schema and stitched_verify
   Observer schema must be designed together to avoid rework on the
   `Command(update=...)` shape. Land them in one commit if possible.
2. **#2** — segment_focus reuses the `system_prompt` + `subject_registry`
   plumbing from #1+#5.
3. **#3** — pure prompt edit; trivial.
4. **#4** — pure prompt edit; trivial.

Bump `AGENT_CODE_VERSION` v5 → v6 on the #1+#5 commit, then v6 → v7 if #3
or #4 land in a separate commit.

---

## What is intentionally NOT in this plan

- **Scan Search tool** (LensWalk's 0.25 fps wide sweep): redundant with our
  existing SigLIP retrieval.
- **Independent Memory Update LLM call** (LensWalk Figure 10): saves API
  cost by piggybacking on Observer responses.
- **Subject Registry → LangMem mirror**: would pollute cross-session memory
  with intra-video entities.
- **Max 20 tool calls / 128-frame stitched cap**: student token budget;
  stay at 8 / 24.
- **VLM swap experiments**: separate decision.

## Critical files

- `app/tools.py` — #1, #2, #5 (new tools, deprecation note, merge helper)
- `app/vqa.py` — #2, #3, #5 (system_prompt arg, observer prompts, registry header)
- `app/graph.py` — #4, #5 (orchestrator prompt PLAN/OBSERVE, GraphState field)
- `app/eval_fingerprint.py` — `AGENT_CODE_VERSION` bumps
- `tests/test_tools_planner.py` — #1, #2
- `tests/test_vqa.py` — #2, #3
- `tests/test_graph_orchestrator.py` — #1, #5 smoke
- `tests/test_subject_registry.py` — #5 (new file)

## Reference: paper anchors

- LensWalk Figure 5 — Reasoner system prompt (basis for #4)
- LensWalk Figure 7 — Scan Search Observer (style for #2 in tone)
- LensWalk Figure 8 — Segment Focus Observer (direct source for #2 prompt)
- LensWalk Figure 9 — Stitched Verify Observer (direct source for #1 prompt)
- LensWalk Figure 10 — Memory Update prompt (adapted into #5 inline delta schema)
- LensWalk Section B.1 — tool budgets (scaled to 1/3 for our token budget)
- LensWalk Section C.3 — failure modes (Static Repetition + Premature
  Conclusion already guarded by existing `_dedup_tool_calls` and
  `verify_grounding`; Evidence Dilution addressed by #3 prompt tightening)
