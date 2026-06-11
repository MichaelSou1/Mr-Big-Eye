#!/usr/bin/env python
"""Spec §11.1 — generate + filter CoT for the regime-2 cell (ChartQA). A STRONG VLM teacher
(32B, reads charts at .80) produces step-by-step arithmetic rationales on ChartQA TRAIN images;
keep only CoTs whose final answer matches gold (consistency filter). Output: SFT records
(image_path, question, cot, answer) for §11.2. Teacher distillation is honest (the 2026-06-08
pivot retired the self-improvement constraint).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from PIL import Image

from app.distill.eval_common import relaxed_match
from app.distill.methods import vlm_answer
import asyncio

COT_SYS = (
    "You are solving a chart question. Read the needed values off the chart, then reason "
    "step by step (show the arithmetic: differences, ratios, sums, comparisons). End with a "
    "line exactly 'ANSWER: <final answer>'. Be concise."
)


def parse_cot_answer(text: str):
    m = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    ans = m.group(1).strip() if m else ""
    cot = text[:m.start()].strip() if m else text.strip()
    return cot, ans


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-base", default="http://127.0.0.1:30001/v1")
    ap.add_argument("--teacher-model", default="Qwen3-VL-32B-Instruct")
    ap.add_argument("--n", type=int, default=800, help="train examples to attempt")
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/chartqa_train_images")
    ap.add_argument("--out", default="data/distill/poc/chartqa_cot_train.jsonl")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download
    Path(args.img_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq = hf_hub_download("HuggingFaceM4/ChartQA",
                         "data/train-00000-of-00003-49492f364babfa44.parquet", repo_type="dataset")
    df = pd.read_parquet(pq)
    print(f"ChartQA train shard: {len(df)} rows; attempting {args.n}", flush=True)

    kept = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(min(args.n, len(df))):
            row = df.iloc[i]
            q = str(row["query"]); gold = str(row["label"][0] if hasattr(row["label"], "__len__")
                                               and not isinstance(row["label"], str) else row["label"])
            img_path = Path(args.img_dir) / f"train_{i}.png"
            if not img_path.exists():
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                img.save(img_path)
            else:
                img = Image.open(img_path).convert("RGB")
            try:
                out = asyncio.run(vlm_answer(COT_SYS + "\n\nQuestion: " + q, [img], [0.0], temp=0.0,
                                             base_url=args.teacher_base, model=args.teacher_model,
                                             max_tokens=512))
            except Exception as e:
                if (i + 1) % 50 == 0: print(f"  {i+1}: teacher err {str(e)[:60]}", flush=True)
                continue
            cot, ans = parse_cot_answer(out)
            if cot and ans and relaxed_match(ans, gold):
                fh.write(json.dumps({"image_path": str(img_path), "question": q, "cot": cot,
                                     "answer": ans, "gold": gold}, ensure_ascii=False) + "\n")
                fh.flush(); kept += 1
            if (i + 1) % 50 == 0:
                print(f"  attempted {i+1}, kept {kept}", flush=True)
    print(f"DONE: kept {kept} consistent CoTs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
