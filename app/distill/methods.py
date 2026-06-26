"""The comparable methods (Spec §0) with explicit decode control (temp/seed).

free_form runs greedy (temp=0, deterministic); the orchestrated/self-reflect methods
take a seed and run their stochastic critic at temp>0 — this is the variance source the
seed loop characterizes. All reuse the app's correct multimodal payload builder; we only
override temperature/seed before posting.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from PIL import Image

# External orchestrator APIs (e.g. DeepSeek) need the clash proxy; localhost endpoints must
# NOT be proxied. We route per-host explicitly (trust_env=False) so the global NO_PROXY the
# diag scripts set for the localhost VLM doesn't block the external API.
_CLASH_PROXY = os.environ.get("CLASH_PROXY", "http://127.0.0.1:7890")

from app.config import settings
from app.vqa import (
    _build_qa_payload,
    _extract_text,
    _local_endpoint_config,
    _post_json_for_config,
)

CRITIC_TEMP = 0.7   # expose the orchestrator stochasticity the seed loop measures


async def vlm_answer(question, frames, ts, *, temp=0.0, seed=None, max_tokens=512,
                     base_url=None, model=None) -> str:
    """One VLM forward pass at the given temperature/seed. base_url/model override the endpoint
    (used for the sighted-critic 2nd VLM). Retries transient connection drops (vllm under load
    occasionally closes a stream mid-body)."""
    import asyncio
    cfg = _local_endpoint_config()
    if base_url:
        cfg = cfg.__class__(**{**cfg.__dict__, "base_url": base_url, "model_name": model or cfg.model_name})
    payload = _build_qa_payload(question, frames, ts, config=cfg)
    payload["temperature"] = temp
    payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = int(seed)
    # §6 cross-family: InternVL's vLLM chat template can't concatenate a system *string* with a
    # list-content user message (400 "can only concatenate str (not list)"). Fold system → user.
    if "intern" in str(cfg.model_name).lower():
        msgs = payload.get("messages", [])
        sys_txt = "".join(m["content"] for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str))
        rest = [m for m in msgs if m.get("role") != "system"]
        if sys_txt and rest and isinstance(rest[0].get("content"), list):
            rest[0]["content"] = [{"type": "text", "text": sys_txt + "\n\n"}] + rest[0]["content"]
        payload["messages"] = rest
    last = None
    for attempt in range(4):
        try:
            data = await _post_json_for_config(payload, cfg)
            return _extract_text(data, config=cfg)
        except Exception as e:  # transient: RemoteProtocolError, ReadTimeout, 5xx
            last = e
            await asyncio.sleep(1.5 * (attempt + 1))
    raise last


def orch(messages: list[dict], *, temp=CRITIC_TEMP, seed=None, max_tokens=2048) -> str:
    """Text orchestrator call with temp/seed control. max_tokens is generous because the
    orchestrator may be a REASONING model (e.g. deepseek-v4-flash) that emits a long
    reasoning_content before the answer lands in content; too small → empty content."""
    base = settings.orchestrator_api_base_url or settings.vlm_api_base_url
    key = settings.orchestrator_api_key or settings.vlm_api_key
    model = settings.orchestrator_model_name or settings.vlm_model_name
    timeout = settings.orchestrator_api_timeout or settings.vlm_api_timeout
    if not base or not model:
        raise RuntimeError("orchestrator API base URL/model is not configured")
    payload = {"model": model, "messages": messages,
               "temperature": temp, "max_tokens": max_tokens}
    if seed is not None:
        payload["seed"] = int(seed)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    is_local = ("127.0.0.1" in base) or ("localhost" in base)
    ckw: dict[str, Any] = {"timeout": timeout, "trust_env": False}
    direct_hosts = ("xiaomimimo.com",)
    if not is_local and not any(host in base for host in direct_hosts):
        ckw["proxy"] = _CLASH_PROXY     # external API (DeepSeek) via clash
    import time
    last = None
    for attempt in range(4):
        try:
            with httpx.Client(**ckw) as c:
                r = c.post(f"{base.rstrip('/')}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                # reasoning models put the answer in `content` after `reasoning_content`; prefer content.
                return msg.get("content") or ""
        except Exception as e:  # transient API/connection errors
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def _parse_subqs(text: str) -> list[str]:
    import json
    import re
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x) for x in arr][:3] if isinstance(arr, list) else []
    except Exception:
        return []


_CRITIC_SYS = (
    "You are a careful visual-reasoning critic. A vision model answered a multiple-choice "
    "question about video frames. {sight} If its answer may be wrong, list up to 3 specific "
    "visual sub-questions to re-check against the frames (counting, ordering, who-does-what, "
    "fine detail). Output ONLY a JSON array of strings (empty array if clearly reliable)."
)
_INTEGRATE_SYS = ("Integrate the vision model's readings and choose the single best option. "
                  "Reply with the option letter and the option text only.")
_REFLECT = ("Re-examine each provided frame carefully. Check whether every claim in your "
            "previous reasoning is supported by what is visible. Correct any unsupported step, "
            "then give your final answer.")


async def m_free_form(case, frames, ts, seed=0) -> dict[str, Any]:
    a = await vlm_answer(case["question"], frames, ts, temp=0.0)   # greedy
    return {"free_answer": a, "method_answer": a}


async def m_self_reflect(case, frames, ts, seed=0) -> dict[str, Any]:
    q = case["question"]
    free = await vlm_answer(q, frames, ts, temp=0.0)
    cur = free
    for _ in range(2):
        # the reflect turn MUST carry the question + prior answer, else the VLM has no idea
        # what is being asked. (answer_question's history path isn't exposed here, so prefix it.)
        prompt = f"{q}\n\nYour previous answer:\n{cur}\n\n{_REFLECT}"
        cur = await vlm_answer(prompt, frames, ts, temp=CRITIC_TEMP, seed=seed)
    return {"free_answer": free, "method_answer": cur}


async def _orch_loop(case, frames, ts, seed, sighted: bool,
                     critic_base=None, critic_model=None) -> dict[str, Any]:
    q = case["question"]
    free = await vlm_answer(q, frames, ts, temp=0.0)
    sight = ("You CANNOT see the frames; you only see its reading."
             if not sighted else "You can re-examine the frames yourself.")
    critic = orch([{"role": "system", "content": _CRITIC_SYS.format(sight=sight)},
                   {"role": "user", "content": f"QUESTION:\n{q}\n\nVISION MODEL ANSWER:\n{free}"}],
                  seed=seed)
    subqs = _parse_subqs(critic)
    sub_qa = []
    for sq in subqs:
        # blind critic -> the original VLM answers the sub-q; sighted -> the critic VLM does.
        if sighted and critic_base:
            a = await vlm_answer(sq, frames, ts, temp=0.0, base_url=critic_base, model=critic_model)
        else:
            a = await vlm_answer(sq, frames, ts, temp=0.0)
        sub_qa.append(f"Q: {sq}\nA: {a}")
    if sub_qa:
        final = orch([{"role": "system", "content": _INTEGRATE_SYS},
                      {"role": "user", "content":
                       f"QUESTION:\n{q}\n\nINITIAL READING:\n{free}\n\nRE-CHECK Q&A:\n" +
                       "\n\n".join(sub_qa)}], seed=seed)
    else:
        final = free
    return {"free_answer": free, "method_answer": final}


async def m_orch_blind(case, frames, ts, seed=0) -> dict[str, Any]:
    return await _orch_loop(case, frames, ts, seed, sighted=False)


def make_orch_sighted(critic_base_url: str, critic_model: str):
    async def m(case, frames, ts, seed=0):
        return await _orch_loop(case, frames, ts, seed, sighted=True,
                                critic_base=critic_base_url, critic_model=critic_model)
    return m


METHOD_FNS = {
    "free_form": m_free_form,
    "self_reflect": m_self_reflect,
    "orch_reflect_blind": m_orch_blind,
}
