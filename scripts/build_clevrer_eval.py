#!/usr/bin/env python
"""Build a CLEVRER reasoning-MCQ pilot set (EvalCase JSONL + manifest + download
list) from zechen-nlp/clevrer train parquets. CLEVRER = synthetic collision
videos: perception is clean (rendered objects) so failures isolate REASONING —
the "perception ⟂ reasoning" data we need to test internalization headroom.

Reasoning types used: predictive / counterfactual / explanatory (multi-step
physical/causal). Videos come from MIT per-file URLs (no 12GB zip).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LABELS = ["A", "B", "C", "D", "E"]


def format_mcq(question: str, choices: list[str]) -> str:
    block = [question.strip(), "", "Candidates:"]
    for lab, c in zip(LABELS, choices):
        block.append(f"{lab}) {c}")
    block += ["", "Answer the question and reference key frames with [FRAME:t=...] markers."]
    return "\n".join(block)


def main() -> int:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=70)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-dir", default="data/eval/datasets/clevrer_pilot")
    args = ap.parse_args()

    rows = []
    for qt in ["predictive", "counterfactual", "explanatory"]:
        df = pd.read_parquet(hf_hub_download("zechen-nlp/clevrer", f"{qt}/train-00000-of-00001.parquet", repo_type="dataset"))
        for _, r in df.iterrows():
            gpt = [v for f, v in zip(r["conversations"]["from"], r["conversations"]["value"]) if f == "gpt"]
            ans = str(gpt[-1]) if gpt else ""
            choices = [str(c) for c in r["choices"]["choice"]]
            if not ans or len(choices) < 2:
                continue
            rows.append({"video": str(r["video"]), "qid": int(r["question_id"]),
                         "qtype": qt, "question": str(r["question"]),
                         "choices": choices, "answer": ans})

    random.seed(args.seed); random.shuffle(rows)
    seen, picked = set(), []
    # balance across types
    per_type = {qt: 0 for qt in ["predictive", "counterfactual", "explanatory"]}
    cap = args.n_videos // 3 + 2
    for r in rows:
        if r["video"] in seen:
            continue
        if per_type[r["qtype"]] >= cap:
            continue
        seen.add(r["video"]); per_type[r["qtype"]] += 1; picked.append(r)
        if len(picked) >= args.n_videos:
            break

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cases, manifest, dl = [], [], []
    for r in picked:
        vid_name = Path(r["video"]).name              # video_00001.mp4
        idx = int(vid_name.replace("video_", "").replace(".mp4", ""))
        lo = (idx // 1000) * 1000
        subfolder = f"video_{lo:05d}-{lo+1000:05d}"
        url = f"https://data.csail.mit.edu/clevrer/videos/train/{subfolder}/{vid_name}"
        case_id = f"clevrer-{idx}-{r['qid']}"
        cases.append({"case_id": case_id, "video_id": "",
                      "question": format_mcq(r["question"], r["choices"]),
                      "reference_answer": r["answer"], "required_keywords": [],
                      "forbidden_keywords": [], "gold_timestamps": [], "gold_scenes": [],
                      "question_type": r["qtype"]})
        manifest.append({"case_id": case_id, "vidor_path": vid_name, "clevrer_url": url, "qtype": r["qtype"]})
        dl.append((vid_name, url))

    (out / "cases.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "download_list.txt").write_text("\n".join(f"{n}\t{u}" for n, u in dl) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "videos": len(dl), "by_type": per_type}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
