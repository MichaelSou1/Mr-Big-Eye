#!/usr/bin/env python
"""Diagnostic: does tool-free self-reflection over the SAME uniform frames beat
single-forward? (Option B, frame-controlled.)

For each pilot case we feed the SAME uniform-16 frames and compare:
  - free-form: one forward pass, answer.
  - reflection: 2 extra turns of pure self-re-examination (no tools, no new
    frames) — "re-check each claim against the frames, fix errors, finalize".

The decisive number is reflect_right AND free_wrong: the cases whose answer the
4B can only reach by reflecting = the internalizable reflection gap. Also reports
free_right AND reflect_wrong (reflection breaking correct answers).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from PIL import Image

from app.distill.common import read_json, sampler_frame_manifest
from app.distill.filter_consistency import _answer_matches
from app.vqa import LocalVLMBackbone

REFLECT_PROMPT = (
    "Re-examine each provided frame carefully. Check whether every claim in your "
    "previous reasoning is actually supported by what is visible. If any step is "
    "wrong or unsupported, correct it. Then give your final answer."
)


def _load_frames(manifest):
    frames, ts = [], []
    for item in manifest:
        p = Path(str(item.get("path") or ""))
        if p.exists():
            frames.append(Image.open(p).convert("RGB"))
            ts.append(float(item["timestamp"]))
    return frames, ts


async def _reflect(backbone, question, frames, ts, n_turns=2):
    """Free-form answer then n_turns of self-reflection over the SAME frames."""
    a0 = await backbone.answer_question(question, frames, ts)
    history = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": a0},
    ]
    cur = a0
    for _ in range(n_turns):
        cur = await backbone.answer_question(REFLECT_PROMPT, frames, ts, history=history)
        history.append({"role": "user", "content": REFLECT_PROMPT})
        history.append({"role": "assistant", "content": cur})
    return a0, cur


async def main_async() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default="data/distill/pilot/trajectories")
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--output", default="data/distill/pilot/reflection_gap_diag.json")
    args = ap.parse_args()

    backbone = LocalVLMBackbone()
    rows = []
    for path in sorted(Path(args.traj_dir).glob("*.json")):
        traj = read_json(path)
        case = traj.get("case") or {}
        frames, ts = _load_frames(sampler_frame_manifest(traj))
        if not frames:
            continue
        q = str(case.get("question") or "")
        ref = str(case.get("reference_answer") or "")
        free_ans, refl_ans = await _reflect(backbone, q, frames, ts, n_turns=args.turns)
        free_ok = _answer_matches(q, ref, free_ans)
        refl_ok = _answer_matches(q, ref, refl_ans)
        rows.append({
            "case_id": case.get("case_id"), "qtype": case.get("question_type"),
            "free_correct": bool(free_ok), "reflect_correct": bool(refl_ok),
        })
        print(f"{case.get('case_id')}: free={free_ok} reflect={refl_ok} qtype={case.get('question_type')}", flush=True)

    n = len(rows)
    free = sum(r["free_correct"] for r in rows)
    refl = sum(r["reflect_correct"] for r in rows)
    gain = [r for r in rows if r["reflect_correct"] and not r["free_correct"]]
    lost = [r for r in rows if r["free_correct"] and not r["reflect_correct"]]
    summary = {
        "n": n,
        "free_accuracy": free / n if n else 0,
        "reflect_accuracy": refl / n if n else 0,
        "reflect_gain_cases": len(gain),       # reflect-right AND free-wrong (the signal)
        "reflect_lost_cases": len(lost),       # free-right AND reflect-wrong (reflection breaking)
        "net_reflection_gain": (refl - free) / n if n else 0,
        "gain_case_ids": [r["case_id"] for r in gain],
        "gain_by_qtype": dict(collections.Counter(r["qtype"] for r in gain)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
