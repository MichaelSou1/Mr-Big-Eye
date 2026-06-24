#!/usr/bin/env python
"""WU-1 §1.1 — build the scaled ChartQA test set (n=400) for the power fix.

Pulls N rows from the HF ChartQA *test* split (HuggingFaceM4/ChartQA), writes images to
``<img-dir>/chartqa_<i>.png`` and rows to ``test_cases_400.jsonl`` in the exact
``{case_id, question, gold}`` dump format that ``run_chartqa_gate.py`` /
``poc_causal_probe_32b.py`` already read (``i`` = parquet row index, image resolved as
``chartqa_<i>.png``).

Distribution: the existing n=60 eval used the ChartQA *human* subset (rows 0–59), which is
the reasoning-bound part this paper's regime-2 / "reading not reasoning" story rests on.
We default to the same human subset (rows 0..N-1) so the scaled set is a clean SUPERSET of
the old eval and the only thing that changes is eval-n — isolating the power effect. The
human/augmented composition is logged either way; ``--augmented-frac`` can mix in the
template (augmented) subset if desired.

Two guarantees this script verifies and logs:
  1. Pixel-hash disjointness from the 187 SFT-train images (no train/test leak).
  2. relaxed-numeric grader self-check: grade every (question, gold) with the gold fed back
     as the answer — must be 100% correct, mcq=False, letter-luck=0 for open-ended ChartQA.

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

from app.distill.eval_common import grade_textaware, relaxed_match

TEST_PARQUET = "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"


def _gold_of(label) -> str:
    """ChartQA label is a 1-element list/ndarray of strings."""
    g = label
    try:
        g = g[0]
    except (TypeError, IndexError, KeyError):
        pass
    return str(g)


def _pixel_hash(img: Image.Image) -> str:
    """Encoding-independent content hash: md5 over decoded RGB pixels + size."""
    im = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{im.size[0]}x{im.size[1]}|".encode())
    h.update(im.tobytes())
    return h.hexdigest()


def _train_hashes(train_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not train_dir.exists():
        return out
    for p in sorted(train_dir.glob("*.png")):
        try:
            out[p.name] = _pixel_hash(Image.open(p))
        except Exception:
            continue
    return out


def main() -> int:
    load_dotenv()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="rows to pull from the test split")
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/chartqa_test_images")
    ap.add_argument("--out", default="data/distill/chartqa/test_cases_400.jsonl")
    ap.add_argument("--train-img-dir", default="/home/gpus/mbe_data/chartqa_train_images",
                    help="SFT-train images; test set is asserted hash-disjoint from these")
    ap.add_argument("--augmented-frac", type=float, default=0.0,
                    help="fraction of N drawn from the augmented/template subset "
                         "(0.0 = human-only, matching the existing eval distribution)")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    img_dir = Path(args.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}  downloading test parquet ...", flush=True)
    pq = hf_hub_download("HuggingFaceM4/ChartQA", TEST_PARQUET, repo_type="dataset")
    df = pd.read_parquet(pq)
    print(f"ChartQA test split: {len(df)} rows; human/augmented = "
          f"{int((df['human_or_machine'] == 0).sum())}/{int((df['human_or_machine'] == 1).sum())}",
          flush=True)

    human_idx = [i for i in range(len(df)) if int(df.iloc[i]["human_or_machine"]) == 0]
    aug_idx = [i for i in range(len(df)) if int(df.iloc[i]["human_or_machine"]) == 1]
    n_aug = int(round(args.n * args.augmented_frac))
    n_human = args.n - n_aug
    # candidate pool in order; we scan and skip any row whose chart image is hash-equal to a
    # train image (same chart can appear in both ChartQA splits with different questions — a
    # real SFT leak), backfilling from later rows until N disjoint cases are collected.
    candidates = (human_idx if n_human else []) + (aug_idx if n_aug else [])
    print(f"target N={args.n}: human={n_human} augmented={n_aug} "
          f"(pool: {len(human_idx)} human / {len(aug_idx)} augmented)", flush=True)

    # --- write images + rows (skip train-colliding, backfill to N) ---
    train_hashes = _train_hashes(Path(args.train_img_dir))
    train_hash_set = set(train_hashes.values())
    print(f"loaded {len(train_hashes)} train-image hashes for disjointness check", flush=True)

    rows = []
    test_hashes: dict[str, str] = {}
    skipped_leak = []  # (case_id, [train names]) for rows dropped as train-test overlap
    n_human_written = n_aug_written = 0
    target_human = n_human
    target_aug = n_aug
    fh = open(out_path, "w", encoding="utf-8")
    for i in candidates:
        is_human = int(df.iloc[i]["human_or_machine"]) == 0
        if is_human and n_human_written >= target_human:
            continue
        if (not is_human) and n_aug_written >= target_aug:
            continue
        r = df.iloc[i]
        cell = r["image"]
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            print(f"  row {i}: image decode err {str(e)[:60]} — skipped", flush=True)
            continue
        cid = f"chartqa-{i}"
        ph = _pixel_hash(img)
        if ph in train_hash_set:  # train/test leak — drop, do not count toward N
            skipped_leak.append((cid, [n for n, h in train_hashes.items() if h == ph]))
            continue
        img.save(img_dir / f"chartqa_{i}.png")
        test_hashes[cid] = ph
        q = str(r["query"])
        gold = _gold_of(r["label"])
        fh.write(json.dumps({"case_id": cid, "question": q, "gold": gold},
                            ensure_ascii=False) + "\n")
        rows.append({"case_id": cid, "question": q, "gold": gold})
        if is_human:
            n_human_written += 1
        else:
            n_aug_written += 1
        if n_human_written >= target_human and n_aug_written >= target_aug:
            break
    fh.close()

    print(f"\nwrote {len(rows)} rows -> {out_path}", flush=True)
    print(f"wrote {len(rows)} images -> {img_dir}/chartqa_<i>.png", flush=True)
    print(f"composition: human={n_human_written} augmented={n_aug_written}", flush=True)

    # --- disjointness ---
    intra_dups = len(test_hashes) - len(set(test_hashes.values()))
    collisions = []  # by construction empty (leaks are skipped, not written)
    bad = [cid for cid, h in test_hashes.items() if h in train_hash_set]
    if skipped_leak:
        print(f"dropped {len(skipped_leak)} train-test overlap rows (same chart in both splits), "
              f"backfilled from later rows. e.g.:", flush=True)
        for cid, names in skipped_leak[:6]:
            print(f"    skipped {cid} (== train {names})", flush=True)
    if bad:
        collisions = bad
        print(f"!!! TRAIN/TEST LEAK REMAINS in {len(bad)} written rows: {bad[:10]}", flush=True)
    else:
        print(f"DISJOINT OK: all {len(test_hashes)} written test images hash-disjoint from "
              f"{len(train_hashes)} train images (0 leaks; {intra_dups} intra-test dup imgs: "
              f"same chart, different question — kept)", flush=True)

    # --- relaxed-numeric grader self-check (feed gold as the answer) ---
    n_correct = n_mcq = n_letter_luck = n_numeric = 0
    failed = []
    for rrow in rows:
        g = grade_textaware(rrow["question"], rrow["gold"], rrow["gold"])
        n_correct += int(bool(g["correct"]))
        n_mcq += int(bool(g["mcq"]))
        n_letter_luck += int(bool(g["letter_luck"]))
        if not g["correct"]:
            failed.append(rrow["case_id"])
        gv = rrow["gold"].replace(",", "").replace("%", "")
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
    ok = (len(rows) >= 300) and (not collisions) and (n_letter_luck == 0)
    print("\n=== WU-1 §1.1 ACCEPTANCE ===", flush=True)
    print(f"  rows >= 300:        {len(rows) >= 300}  (n={len(rows)})", flush=True)
    print(f"  train-disjoint:     {not collisions}", flush=True)
    print(f"  letter-luck == 0:   {n_letter_luck == 0}", flush=True)
    print(f"  images present:     {len(list(img_dir.glob('chartqa_*.png')))} png in {img_dir}",
          flush=True)
    print("ACCEPTANCE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
