#!/usr/bin/env python
"""WU-3 — build the TabMWP test set (mirror of ``build_chartqa_test.py``) for the second
reasoning-bound dataset, to kill the n=1-dataset reviewer attack.

Source: ``zyhang1998/tabmwp`` (canonical TabMWP, rendered TABLE images embedded as PNG bytes),
``problems_test.parquet`` rows ``{id, Prompt, Answer, Choices, Image bytes}``. We keep the
**free-response numeric** subset (``Choices == []`` and a numeric ``Answer``) — the exact open-ended
substrate ChartQA uses, so there are NO letter options → no disguised-accuracy / letter-luck
confound (answers 2402.14897), and the relaxed-numeric grader applies unchanged. TabMWP is the
right complement to ChartQA: the table is **re-readable** (values live in cells) yet the answer
needs **multi-step arithmetic** (sum / diff / stem-and-leaf counts), so it directly tests whether
"reading, not reasoning" replicates on a second chart/table regime.

Output mirrors the ChartQA dump format the downstream harness already reads
(``run_chartqa_gate.py`` / ``eval_sft_n400.py`` / ``battery_n400.py`` with ``--img-prefix tabmwp``):
  - rows  -> ``data/distill/tabmwp/test_cases_400.jsonl`` as ``{case_id:"tabmwp-<id>", question, gold}``
  - images-> ``<img-dir>/tabmwp_<id>.png`` (``idx`` = TabMWP problem id, image = ``tabmwp_<idx>.png``)

Two guarantees this script verifies and logs (same as the ChartQA builder):
  1. Pixel-hash disjointness from the TabMWP *train* table images (no SFT train/test leak). The
     train hashes are computed self-contained from ``problems_train.parquet`` (no need to save them).
  2. relaxed-numeric grader self-check: grade every (question, gold) with the gold fed back as the
     answer — must be 100% correct, mcq=False, letter-luck=0 for this open-ended subset.

Download uses HF_ENDPOINT (hf-mirror) from .env. Run in env `mbe-up`.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from PIL import Image

from app.distill.eval_common import grade_textaware

HF_REPO = "zyhang1998/tabmwp"
TEST_PARQUET = "problems_test.parquet"
TRAIN_PARQUET = "problems_train.parquet"


def _is_free_numeric(choices, answer) -> bool:
    """Keep only free-response (no multiple-choice options) numeric-answer questions."""
    try:
        if choices is not None and len(choices) > 0:
            return False
    except TypeError:
        pass
    s = str(answer).replace(",", "").replace("%", "").replace("$", "").strip()
    return bool(re.fullmatch(r"-?\d+\.?\d*", s))


def _pixel_hash(img: Image.Image) -> str:
    """Encoding-independent content hash: md5 over decoded RGB pixels + size."""
    im = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{im.size[0]}x{im.size[1]}|".encode())
    h.update(im.tobytes())
    return h.hexdigest()


def _img_bytes(cell):
    """The image cell is raw PNG bytes (or a {'bytes': ...} dict, depending on parquet writer)."""
    if isinstance(cell, dict):
        return cell.get("bytes")
    return cell


def main() -> int:
    load_dotenv()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="free-numeric rows to keep from the test split")
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/tabmwp_test_images")
    ap.add_argument("--out", default="data/distill/tabmwp/test_cases_400.jsonl")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    img_dir = Path(args.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}  downloading TabMWP parquets ...", flush=True)
    test_pq = hf_hub_download(HF_REPO, TEST_PARQUET, repo_type="dataset")
    train_pq = hf_hub_download(HF_REPO, TRAIN_PARQUET, repo_type="dataset")
    df = pd.read_parquet(test_pq)
    dft = pd.read_parquet(train_pq)
    n_mc = int(df["Choices"].apply(lambda c: bool(c is not None and len(c) > 0)).sum())
    print(f"TabMWP test split: {len(df)} rows ({n_mc} multi-choice / {len(df) - n_mc} free-response); "
          f"train split: {len(dft)} rows", flush=True)

    # --- self-contained train/test disjointness: hash EVERY train table image ---
    train_hashes: dict[str, str] = {}  # pid -> hash
    for i in range(len(dft)):
        raw = _img_bytes(dft.iloc[i]["Image bytes"])
        if raw is None:
            continue
        try:
            train_hashes[str(dft.iloc[i]["id"])] = _pixel_hash(Image.open(io.BytesIO(raw)))
        except Exception:
            continue
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(dft)} train images", flush=True)
    train_hash_set = set(train_hashes.values())
    print(f"loaded {len(train_hashes)} train-image hashes for disjointness check", flush=True)

    # --- write images + rows (free-numeric only; skip train-colliding; backfill to N) ---
    rows = []
    test_hashes: dict[str, str] = {}
    skipped_leak = []   # (case_id, [train pids]) dropped as train/test overlap
    n_seen_free = 0
    fh = open(out_path, "w", encoding="utf-8")
    for i in range(len(df)):
        r = df.iloc[i]
        if not _is_free_numeric(r["Choices"], r["Answer"]):
            continue
        n_seen_free += 1
        raw = _img_bytes(r["Image bytes"])
        if raw is None:
            continue
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            print(f"  id {r['id']}: image decode err {str(e)[:60]} — skipped", flush=True)
            continue
        pid = str(r["id"])
        cid = f"tabmwp-{pid}"
        ph = _pixel_hash(img)
        if ph in train_hash_set:  # train/test leak — drop, do not count toward N
            skipped_leak.append((cid, [p for p, h in train_hashes.items() if h == ph]))
            continue
        img.save(img_dir / f"tabmwp_{pid}.png")
        test_hashes[cid] = ph
        q = str(r["Prompt"])
        gold = str(r["Answer"])
        fh.write(json.dumps({"case_id": cid, "question": q, "gold": gold},
                            ensure_ascii=False) + "\n")
        rows.append({"case_id": cid, "question": q, "gold": gold})
        if len(rows) >= args.n:
            break
    fh.close()

    print(f"\nwrote {len(rows)} rows -> {out_path}", flush=True)
    print(f"wrote {len(rows)} images -> {img_dir}/tabmwp_<id>.png "
          f"(scanned {n_seen_free} free-numeric rows)", flush=True)

    # --- disjointness ---
    intra_dups = len(test_hashes) - len(set(test_hashes.values()))
    bad = [cid for cid, h in test_hashes.items() if h in train_hash_set]
    if skipped_leak:
        print(f"dropped {len(skipped_leak)} train-test overlap rows (same table image in both "
              f"splits), backfilled from later rows. e.g.:", flush=True)
        for cid, pids in skipped_leak[:6]:
            print(f"    skipped {cid} (== train pid {pids})", flush=True)
    if bad:
        print(f"!!! TRAIN/TEST LEAK REMAINS in {len(bad)} written rows: {bad[:10]}", flush=True)
    else:
        print(f"DISJOINT OK: all {len(test_hashes)} written test images hash-disjoint from "
              f"{len(train_hashes)} train images (0 leaks; {intra_dups} intra-test dup imgs)", flush=True)

    # --- relaxed-numeric grader self-check (feed gold as the answer) ---
    n_correct = n_mcq = n_letter_luck = n_numeric = 0
    failed = []
    for rrow in rows:
        g = grade_textaware(rrow["question"], rrow["gold"], rrow["gold"])
        n_correct += int(bool(g["correct"]))
        n_mcq += int(bool(g["mcq"]))
        n_letter_luck += int(bool(g["letter_luck"]))
        if not g["correct"]:
            failed.append((rrow["case_id"], rrow["gold"]))
        gv = rrow["gold"].replace(",", "").replace("%", "").replace("$", "")
        if re.fullmatch(r"-?\d+\.?\d*", gv):
            n_numeric += 1
    print("\n=== grader self-check (gold-as-answer) ===", flush=True)
    print(f"  grade_textaware correct: {n_correct}/{len(rows)}  "
          f"(numeric golds={n_numeric}, text golds={len(rows) - n_numeric})", flush=True)
    print(f"  mcq questions: {n_mcq}   letter-luck: {n_letter_luck}  (open-ended expects 0)",
          flush=True)
    if failed:
        print(f"  WARN: {len(failed)} golds did not self-grade correct: {failed[:8]}", flush=True)

    # --- acceptance ---
    ok = (len(rows) >= 300) and (not bad) and (n_letter_luck == 0) and (n_mcq == 0)
    print("\n=== WU-3 build_tabmwp_test ACCEPTANCE ===", flush=True)
    print(f"  rows >= 300:        {len(rows) >= 300}  (n={len(rows)})", flush=True)
    print(f"  train-disjoint:     {not bad}", flush=True)
    print(f"  no mcq / letter-luck: {n_mcq == 0 and n_letter_luck == 0}", flush=True)
    print(f"  images present:     {len(list(img_dir.glob('tabmwp_*.png')))} png in {img_dir}",
          flush=True)
    print("ACCEPTANCE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
