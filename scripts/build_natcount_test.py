#!/usr/bin/env python
"""N3 §build — natural-image counting test set (the "V*-style / small-target,
non-rereadable natural-image pole").

We operationalize the natural-image regime (small targets that the VLM physically
cannot cheaply re-perceive — the mirror of 2510.23482) with **TallyQA *complex*
counting** on Visual-Genome natural images. Rationale:

  * Natural cluttered scenes; the answer is a *filtered* count
    ("how many people are wearing blue shirts?") => perceive + filter + count,
    the natural-image analog of charts' perceive + arithmetic. The textual
    reasoning chain is non-trivial, so "is the CoT load-bearing?" is meaningful.
  * Free-form **integer** answers => same relaxed-numeric grader and the SAME
    corrupt-number / snap-follow-other probe as ChartQA/TabMWP. NO multiple choice
    => no letter-luck / disguised-accuracy (2402.14897), consistent substrate.
  * `is_simple=False` isolates the reasoning-bound subset (simple = single salient
    object, pure perception with no chain — excluded).

Real V*Bench is multiple-choice (attribute / spatial-relation), incompatible with
the numeric corrupt-number probe; TallyQA-complex preserves the *scientific*
property (small targets in natural images, hard to re-perceive) while keeping the
probe machinery and grader identical across all three regimes. The limitation
(VG median ~500px, not 4K-tiny-target) is recorded in the prereg/snapshot.

Selection (fixed BEFORE results, see docs/snapshots/n3_prereg_0625.md):
  complex (is_simple=False) AND integer answer >= MIN_ANS, <= MAX_QA_PER_IMG per
  image, scanning test shards in order until N collected. Image-hash dup count is
  logged. No train set exists (N3 is a base-model probe, no SFT) => no disjointness
  check needed.

Writes images to ``<img-dir>/natcount_<i>.png`` and rows to
``{case_id:"natcount-<i>", question, gold}`` — the exact dump format
``battery_n400.py`` reads (case_id prefix -> image filename prefix).

Uses the official HF endpoint (hf-mirror's 308 redirect breaks huggingface_hub's
metadata HEAD; huggingface.co is directly reachable here). Run in env `mbe-up`.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from app.distill.eval_common import grade_textaware

REPO = "vikhyatk/tallyqa-test"
N_SHARDS = 19
SHARD = "data/test-{:05d}-of-00019.parquet"


def _pixel_hash(img: Image.Image) -> str:
    im = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{im.size[0]}x{im.size[1]}|".encode())
    h.update(im.tobytes())
    return h.hexdigest()


def main() -> int:
    # hf-mirror's 308 redirect trips huggingface_hub's metadata check; use the
    # official endpoint (reachable here) and no token (public dataset).
    os.environ.pop("HF_ENDPOINT", None)
    os.environ.pop("HF_TOKEN", None)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--min-ans", type=int, default=2,
                    help="keep complex counts >= this (non-degenerate; harder perception)")
    ap.add_argument("--max-ans", type=int, default=20, help="drop absurd outliers")
    ap.add_argument("--max-qa-per-img", type=int, default=2,
                    help="cap cases sharing one image (diversity)")
    ap.add_argument("--max-shards", type=int, default=8)
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/natcount_test_images")
    ap.add_argument("--out", default="data/distill/natcount/test_cases_400.jsonl")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    img_dir = Path(args.img_dir); img_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    test_hashes: dict[str, str] = {}
    ans_dist = Counter()
    src_dist = Counter()
    n_complex_seen = n_simple_seen = 0
    fh = open(out_path, "w", encoding="utf-8")
    case_i = 0
    for shard in range(min(args.max_shards, N_SHARDS)):
        if len(rows) >= args.n:
            break
        fn = SHARD.format(shard)
        print(f"[shard {shard}] downloading {fn} ...", flush=True)
        p = hf_hub_download(REPO, fn, repo_type="dataset")
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            if len(rows) >= args.n:
                break
            qa_list = list(row["qa"])
            per_img = 0
            img = None
            for qa in qa_list:
                if len(rows) >= args.n or per_img >= args.max_qa_per_img:
                    break
                is_simple = bool(qa["is_simple"])
                if is_simple:
                    n_simple_seen += 1
                    continue
                n_complex_seen += 1
                a = str(qa["answer"]).strip()
                if not re.fullmatch(r"\d+", a):
                    continue
                av = int(a)
                if av < args.min_ans or av > args.max_ans:
                    continue
                if img is None:
                    cell = row["image"]
                    raw = cell["bytes"] if isinstance(cell, dict) else cell
                    try:
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                    except Exception as e:
                        print(f"  image decode err {str(e)[:50]} — skip image", flush=True)
                        break
                cid = f"natcount-{case_i}"
                img.save(img_dir / f"natcount_{case_i}.png")
                test_hashes[cid] = _pixel_hash(img)
                q = str(qa["question"]).strip()
                fh.write(json.dumps({"case_id": cid, "question": q, "gold": a},
                                    ensure_ascii=False) + "\n")
                rows.append({"case_id": cid, "question": q, "gold": a})
                ans_dist[av] += 1
                src_dist[str(qa.get("data_source", "?"))] += 1
                per_img += 1
                case_i += 1
    fh.close()

    print(f"\nscanned: complex={n_complex_seen} simple={n_simple_seen}", flush=True)
    print(f"wrote {len(rows)} rows -> {out_path}", flush=True)
    print(f"wrote {len(rows)} images -> {img_dir}/natcount_<i>.png", flush=True)
    n_uniq_img = len(set(test_hashes.values()))
    print(f"unique images: {n_uniq_img} (cases sharing an image: {len(rows) - n_uniq_img})", flush=True)
    print(f"answer dist: {dict(sorted(ans_dist.items()))}", flush=True)
    print(f"data_source dist: {dict(src_dist)}", flush=True)

    # --- relaxed-numeric grader self-check (feed gold as the answer) ---
    n_correct = n_mcq = n_letter_luck = 0
    failed = []
    for rrow in rows:
        g = grade_textaware(rrow["question"], rrow["gold"], rrow["gold"])
        n_correct += int(bool(g["correct"]))
        n_mcq += int(bool(g["mcq"]))
        n_letter_luck += int(bool(g["letter_luck"]))
        if not g["correct"]:
            failed.append(rrow["case_id"])
    print("\n=== grader self-check (gold-as-answer) ===", flush=True)
    print(f"  grade_textaware correct: {n_correct}/{len(rows)}", flush=True)
    print(f"  mcq: {n_mcq}   letter-luck: {n_letter_luck}  (open-ended expects 0)", flush=True)
    if failed:
        print(f"  WARN: {len(failed)} golds did not self-grade: {failed[:8]}", flush=True)

    ok = (len(rows) >= 300) and (n_mcq == 0) and (n_letter_luck == 0)
    print("\n=== N3 build ACCEPTANCE ===", flush=True)
    print(f"  rows >= 300:      {len(rows) >= 300}  (n={len(rows)})", flush=True)
    print(f"  mcq == 0:         {n_mcq == 0}", flush=True)
    print(f"  letter-luck == 0: {n_letter_luck == 0}", flush=True)
    print(f"  images present:   {len(list(img_dir.glob('natcount_*.png')))} png in {img_dir}", flush=True)
    print("ACCEPTANCE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
