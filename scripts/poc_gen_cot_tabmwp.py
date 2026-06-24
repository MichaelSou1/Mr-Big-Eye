#!/usr/bin/env python
"""WU-3 (mirror of ``poc_gen_cot.py``) — generate + consistency-filter teacher CoT for the
TabMWP regime-2 cell. The strong 32B VLM teacher reads TabMWP TRAIN table images and produces
step-by-step arithmetic rationales; keep only CoTs whose final answer matches gold. Output: SFT
records ``{image_path, question, cot, answer, gold}`` for the §11 SFT.

Source: ``zyhang1998/tabmwp`` (``problems_train.parquet``), free-response numeric subset only —
the same open-ended substrate as the test set (built by ``build_tabmwp_test.py``). Train images
are written to ``--img-dir`` as ``tabmwp_<id>.png`` (train/test pids are disjoint by construction).
Run in env `mbe-up`; teacher served at ``--teacher-base`` (32B @ :30001).
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

HF_REPO = "zyhang1998/tabmwp"
TRAIN_PARQUET = "problems_train.parquet"

COT_SYS = (
    "You are solving a question about the table shown. Read the needed values off the table, then "
    "reason step by step (show the arithmetic: differences, sums, means, ratios, stem-and-leaf "
    "counts). End with a line exactly 'ANSWER: <final answer>'. Be concise."
)


def _is_free_numeric(choices, answer) -> bool:
    try:
        if choices is not None and len(choices) > 0:
            return False
    except TypeError:
        pass
    s = str(answer).replace(",", "").replace("%", "").replace("$", "").strip()
    return bool(re.fullmatch(r"-?\d+\.?\d*", s))


def _img_bytes(cell):
    if isinstance(cell, dict):
        return cell.get("bytes")
    return cell


def parse_cot_answer(text: str):
    m = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    ans = m.group(1).strip() if m else ""
    cot = text[:m.start()].strip() if m else text.strip()
    return cot, ans


def main() -> int:
    load_dotenv()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-base", default="http://127.0.0.1:30001/v1")
    ap.add_argument("--teacher-model", default="Qwen3-VL-32B-Instruct")
    ap.add_argument("--n", type=int, default=500, help="train examples to attempt")
    ap.add_argument("--target", type=int, default=160, help="stop once this many consistent CoTs kept")
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/tabmwp_train_images")
    ap.add_argument("--out", default="data/distill/poc/tabmwp_cot_train.jsonl")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download
    Path(args.img_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq = hf_hub_download(HF_REPO, TRAIN_PARQUET, repo_type="dataset")
    df = pd.read_parquet(pq)
    print(f"TabMWP train shard: {len(df)} rows; attempting up to {args.n} free-numeric, "
          f"target {args.target} kept", flush=True)

    kept = attempted = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(len(df)):
            if attempted >= args.n or kept >= args.target:
                break
            row = df.iloc[i]
            if not _is_free_numeric(row["Choices"], row["Answer"]):
                continue
            attempted += 1
            pid = str(row["id"])
            q = str(row["Prompt"]); gold = str(row["Answer"])
            img_path = Path(args.img_dir) / f"tabmwp_{pid}.png"
            if not img_path.exists():
                raw = _img_bytes(row["Image bytes"])
                if raw is None:
                    continue
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                img.save(img_path)
            else:
                img = Image.open(img_path).convert("RGB")
            try:
                out = asyncio.run(vlm_answer(COT_SYS + "\n\nQuestion: " + q, [img], [0.0], temp=0.0,
                                             base_url=args.teacher_base, model=args.teacher_model,
                                             max_tokens=512))
            except Exception as e:
                if attempted % 50 == 0:
                    print(f"  attempt {attempted}: teacher err {str(e)[:60]}", flush=True)
                continue
            cot, ans = parse_cot_answer(out)
            if cot and ans and relaxed_match(ans, gold):
                fh.write(json.dumps({"image_path": str(img_path), "question": q, "cot": cot,
                                     "answer": ans, "gold": gold}, ensure_ascii=False) + "\n")
                fh.flush(); kept += 1
            if attempted % 50 == 0:
                print(f"  attempted {attempted}, kept {kept}", flush=True)
    print(f"DONE: kept {kept} consistent CoTs (attempted {attempted}) -> {args.out}")
    return 0 if kept >= 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())
