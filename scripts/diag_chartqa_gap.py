#!/usr/bin/env python
"""Mechanism test (image, perception-easy / reasoning-hard): does 30B-orchestrated
reflection over 8B reads beat 8B free-form on ChartQA? Reading chart values is
easy for the VLM; multi-step arithmetic is the reasoning. If orch ≫ free here,
the orchestrator-distillation mechanism has real headroom (independent of the
video framing). Single image fed as a one-frame "video".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Local VLM/orchestrator endpoints are on localhost — a clash/HTTP proxy in the
# env routes them away and returns 502. Force-bypass proxy for this process.
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from app.vqa import LocalVLMBackbone
from scripts.diag_orch_reflection import run_case


def relaxed_match(pred: str, gold: str) -> bool:
    """ChartQA relaxed accuracy: 5% tolerance for numbers, normalized exact for text."""
    def norm(s):
        return re.sub(r"[^a-z0-9.]", "", str(s).lower())
    g = str(gold).strip()
    # try numeric
    pm = re.findall(r"-?\d+\.?\d*", pred.replace(",", ""))
    try:
        gv = float(g.replace(",", "").replace("%", ""))
        for p in pm:
            if abs(float(p) - gv) <= abs(gv) * 0.05 + 1e-6:
                return True
        return False
    except ValueError:
        gn = norm(g)
        return bool(gn) and gn in norm(pred)


async def main_async() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/chartqa_images")
    ap.add_argument("--output", default="data/distill/chartqa/gap_diag_8b.json")
    args = ap.parse_args()

    import io
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image
    pq = hf_hub_download("HuggingFaceM4/ChartQA",
                         "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet", repo_type="dataset")
    df = pd.read_parquet(pq)
    Path(args.img_dir).mkdir(parents=True, exist_ok=True)

    bb = LocalVLMBackbone()
    rows = []
    for i in range(min(len(df), args.n * 3)):
        if len(rows) >= args.n:
            break
        r = df.iloc[i]
        q = str(r["query"]); gold = r["label"]
        gold = gold[0] if isinstance(gold, (list, tuple)) else gold
        imgcell = r["image"]
        raw = imgcell["bytes"] if isinstance(imgcell, dict) else imgcell
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            continue
        p = Path(args.img_dir) / f"chartqa_{i}.png"
        img.save(p)
        frames = [img]
        try:
            free, final, nsub = await run_case(bb, q, str(gold), frames, [0.0])
        except Exception as e:
            print(f"{i}: ERROR {e}", flush=True); continue
        fok = relaxed_match(free, gold); ook = relaxed_match(final, gold)
        rows.append({"i": i, "free_correct": bool(fok), "orch_correct": bool(ook), "n_subq": nsub})
        print(f"{i}: free={fok} orch={ook} subq={nsub} gold={gold}", flush=True)

    n = len(rows)
    free = sum(r["free_correct"] for r in rows); orch = sum(r["orch_correct"] for r in rows)
    gain = sum(1 for r in rows if r["orch_correct"] and not r["free_correct"])
    lost = sum(1 for r in rows if r["free_correct"] and not r["orch_correct"])
    summary = {"n": n, "free_accuracy": free / n if n else 0, "orch_accuracy": orch / n if n else 0,
               "orch_gain_cases": gain, "orch_lost_cases": lost, "net_gain": (orch - free) / n if n else 0}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print("\n=== SUMMARY ===\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
