"""Prompt + agent-code fingerprint for the prediction cache key.

Kept in a tiny standalone module so `app.eval_harness` stays importable
without dragging in LangChain (graph.py imports it lazily here).
"""
from __future__ import annotations

import hashlib

# Bump when the orchestrator's *runtime* behavior changes in a way that should
# invalidate cached predictions even if the prompt strings are unchanged
# (e.g. Phase D adds tool-call dedup or force-terminate guards).
AGENT_CODE_VERSION = "v22"


def prompt_fingerprint() -> str:
    """Short, stable hash of the two system prompts the agent uses.

    Editing QA_SYSTEM_PROMPT or _orchestrator_prompt changes the hash and
    therefore invalidates affected prediction-cache entries automatically.
    """
    from app.vqa import QA_SYSTEM_PROMPT
    try:
        from app.graph import _orchestrator_prompt

        orch_video = _orchestrator_prompt(has_video=True)
        orch_novideo = _orchestrator_prompt(has_video=False)
    except Exception:
        # The video-agent orchestrator (app.graph) drags in the langgraph/langchain stack,
        # which the upgrade env (mbe-up) deliberately omits. The ChartQA/probe harness uses
        # the DeepSeek critic in app.distill.methods, not this graph, so its prompt is not
        # part of those runs — fall back to a stable placeholder so the fingerprint stays
        # deterministic without that stack (product env keeps the original hash).
        orch_video = orch_novideo = "<orchestrator-prompt-unavailable>"

    payload = "\x1e".join(
        [
            QA_SYSTEM_PROMPT,
            orch_video,
            orch_novideo,
            AGENT_CODE_VERSION,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
