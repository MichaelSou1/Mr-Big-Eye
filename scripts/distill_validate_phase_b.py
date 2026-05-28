#!/usr/bin/env python
"""Phase B acceptance check (spec §5.8).

Validates the produced train/val sets: structural sanity of each sample (system
first, assistant last, tools present), target-kind & per-tool target coverage,
no leaked base64, no forced-call targets, and Qwen-tokenizer sequence-length
distribution (skipped gracefully if transformers/tokenizer unavailable).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.distill_format import target_kind, target_tool_names
from app.distill_trajectory import FORCED_CALL_IDS


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _structural_errors(sample: dict) -> list[str]:
    errs = []
    msgs = sample.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        errs.append("first_not_system")
    if not msgs or msgs[-1].get("role") != "assistant":
        errs.append("last_not_assistant")
    if "tools" not in sample:
        errs.append("no_tools")
    last = msgs[-1] if msgs else {}
    for tc in last.get("tool_calls") or []:
        if tc.get("id") in FORCED_CALL_IDS:
            errs.append("forced_target")
    if "image_b64" in json.dumps(msgs) or "<omitted_b64" in json.dumps(msgs):
        pass  # placeholders allowed; raw base64 length checked below
    return errs


def _seq_len_stats(samples: list[dict], tokenizer_path: str | None):
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"tokenizer unavailable: {exc}"}
    lengths = []
    for s in samples:
        try:
            text = tok.apply_chat_template(
                s["messages"], tools=s.get("tools"), tokenize=False, add_generation_prompt=False
            )
            lengths.append(len(tok(text)["input_ids"]))
        except Exception:  # noqa: BLE001
            continue
    if not lengths:
        return {"error": "no renderable samples"}
    lengths.sort()
    def pct(p):
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))]
    return {"n": len(lengths), "max": max(lengths), "p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/distillation/train.jsonl")
    parser.add_argument("--val", default="data/distillation/val.jsonl")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer path/name for seq-len stats (e.g. Qwen2.5-7B-Instruct).")
    parser.add_argument("--out", default="data/distillation/phase_b_report.md")
    args = parser.parse_args()

    train = _read_jsonl(Path(args.train)) if Path(args.train).exists() else []
    val = _read_jsonl(Path(args.val)) if Path(args.val).exists() else []
    all_samples = train + val

    errors = Counter()
    for s in all_samples:
        for e in _structural_errors(s):
            errors[e] += 1
    raw_b64 = sum(1 for s in all_samples if '"image_b64": "' in json.dumps(s, ensure_ascii=False) and "omitted" not in json.dumps(s, ensure_ascii=False))

    kind = Counter(target_kind(s) for s in all_samples)
    tools = Counter()
    for s in all_samples:
        for name in target_tool_names(s):
            tools[name] += 1

    seq = _seq_len_stats(all_samples, args.tokenizer)

    ok = not errors and raw_b64 == 0
    lines = [
        "# Phase B report — SFT dataset",
        "",
        f"- train samples: **{len(train)}**, val samples: **{len(val)}**, total **{len(all_samples)}**",
        f"- structural errors: `{dict(errors)}`  (ok={not errors})",
        f"- raw (un-truncated) base64 samples: **{raw_b64}** (must be 0)",
        f"- target kind distribution: `{dict(kind)}`",
        f"- per-tool target counts: `{dict(tools)}`",
        f"- sequence length (qwen tokenizer): `{seq}`",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
