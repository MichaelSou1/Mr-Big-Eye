#!/usr/bin/env python
"""CPU-only posthoc summaries for the TabMWP Full-SFT battery.

The full TabMWP rerun writes per-case forced-continuation answers under
``details[].answers``. This script classifies shuffle/filler/paraphrase answers
as snap-to-gold vs other so P1-8 can be updated without manual JSON spelunking.
It is safe to run before the full battery exists; missing files are reported as
pending and no model is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re as _re
from pathlib import Path
from typing import Any


DEFAULT_PRESENT = Path("data/distill/poc/battery_full8b_tabmwp_present.json")
DEFAULT_MASKED = Path("data/distill/poc/battery_full8b_tabmwp_masked.json")
DEFAULT_JSON = Path("data/distill/poc/full8b_tabmwp_battery_posthoc.json")
DEFAULT_MD = Path("docs/reviews/full8b_tabmwp_battery_posthoc.md")
REQUIRED_FLAT_VARIANTS = ["shuffle", "filler", "paraphrase", "corrupt"]


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = _re.sub(r"[^a-z0-9一-鿿]+", " ", text)
    return _re.sub(r"\s+", " ", text).strip()


def relaxed_match(pred: str, gold: str) -> bool:
    g = str(gold).strip()
    nums = _re.findall(r"-?\d+\.?\d*", str(pred).replace(",", ""))
    try:
        gv = float(g.replace(",", "").replace("%", ""))
        return any(abs(float(p) - gv) <= abs(gv) * 0.05 + 1e-6 for p in nums)
    except ValueError:
        gn = normalize_text(g)
        return bool(gn) and gn in normalize_text(pred)


def pct(x: Any) -> str:
    return "NA" if x is None else f"{100 * float(x):.1f}%"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": st.st_size,
        "sha256": h.hexdigest(),
    }


def classify_flat(details: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n = snap = same_base = follows_injected = other = 0
    examples: list[dict[str, Any]] = []
    for row in details:
        ans = (row.get("answers") or {}).get(key)
        if ans is None:
            continue
        n += 1
        is_snap = relaxed_match(ans, row.get("gold", ""))
        is_same_base = relaxed_match(ans, row.get("base_ans", ""))
        injected = row.get("injected")
        is_follow = bool(injected) and relaxed_match(ans, injected)
        if is_snap:
            snap += 1
        elif is_follow:
            follows_injected += 1
        else:
            other += 1
        if len(examples) < 5 and not is_snap:
            examples.append({
                "case_id": row.get("case_id"),
                "gold": row.get("gold"),
                "base_ans": row.get("base_ans"),
                "injected": injected,
                "answer": ans,
                "same_as_base": is_same_base,
                "follows_injected": is_follow,
            })
        if is_same_base:
            same_base += 1
    return {
        "n": n,
        "snap_to_gold": snap,
        "same_as_base": same_base,
        "follows_injected": follows_injected,
        "other": other,
        "snap_rate": snap / n if n else None,
        "same_base_rate": same_base / n if n else None,
        "follow_rate": follows_injected / n if n else None,
        "other_rate": other / n if n else None,
        "examples_not_snap": examples,
    }


def summarize_file(path: Path) -> dict[str, Any]:
    source = file_fingerprint(path)
    obj = read_json(path)
    if obj is None:
        return {"path": str(path), "source": source, "status": "PENDING", "reason": "missing file"}
    details = obj.get("details") or []
    has_answers = bool(details and (details[0].get("answers") is not None))
    if not has_answers:
        return {
            "path": str(path),
            "source": source,
            "status": "PENDING",
            "reason": "details[].answers missing; rerun with current scripts/battery_n400.py",
            "summary": obj.get("summary") or {},
        }
    interventions = (obj.get("summary") or {}).get("interventions") or {}
    variants = [k for k in ["shuffle", "filler", "paraphrase", "corrupt"] if k in interventions]
    flat_variants = {k: classify_flat(details, k) for k in variants}
    missing_summary_variants = [k for k in REQUIRED_FLAT_VARIANTS if k not in interventions]
    empty_answer_variants = [k for k in REQUIRED_FLAT_VARIANTS
                             if k in interventions and (flat_variants.get(k) or {}).get("n", 0) == 0]
    status = "PASS"
    reason = None
    if missing_summary_variants or empty_answer_variants:
        status = "PENDING"
        parts = []
        if missing_summary_variants:
            parts.append("summary missing variants: " + ",".join(missing_summary_variants))
        if empty_answer_variants:
            parts.append("details[].answers missing variants: " + ",".join(empty_answer_variants))
        reason = "; ".join(parts)
    return {
        "path": str(path),
        "source": source,
        "status": status,
        "reason": reason,
        "summary": obj.get("summary") or {},
        "flat_variants": flat_variants,
        "missing_summary_variants": missing_summary_variants,
        "empty_answer_variants": empty_answer_variants,
    }


def render_md(result: dict[str, Any]) -> str:
    lines = [
        "# TabMWP Full-SFT Battery Posthoc",
        "",
        "This CPU-only summary classifies forced-continuation answers from `details[].answers`.",
        "Shuffle has no injected value by design, so the paper-facing quantity is snap-to-gold/other, not follow-under-shuffle.",
        "",
        "| cell | status | n_eval | variant | n | snap | same as base | injected-follow | other |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for cell, rec in result["cells"].items():
        status = rec.get("status", "PENDING")
        n_eval = (rec.get("summary") or {}).get("n_eval", "NA")
        flat = rec.get("flat_variants") or {}
        if not flat:
            lines.append(f"| {cell} | {status}: {rec.get('reason', 'NA')} | {n_eval} | NA | NA | NA | NA | NA | NA |")
            continue
        for variant, m in flat.items():
            lines.append(
                f"| {cell} | {status} | {n_eval} | {variant} | {m['n']} | "
                f"{m['snap_to_gold']}/{m['n']} ({pct(m['snap_rate'])}) | "
                f"{m['same_as_base']}/{m['n']} ({pct(m['same_base_rate'])}) | "
                f"{m['follows_injected']}/{m['n']} ({pct(m['follow_rate'])}) | "
                f"{m['other']}/{m['n']} ({pct(m['other_rate'])}) |"
            )
    lines += [
        "",
        "## Cell Status",
        "",
    ]
    for cell, rec in result["cells"].items():
        reason = rec.get("reason") or "OK"
        source = rec.get("source") or {}
        sha = source.get("sha256")
        short_sha = sha[:12] if sha else "NA"
        lines.append(f"- `{cell}`: {rec.get('status', 'PENDING')} ({reason}); source_sha256={short_sha}")
    lines += [
        "",
        "## Notes",
        "",
        "- `same as base` is useful because base answers are correct for kept cases, but `snap` is the stricter paper-facing label.",
        "- `injected-follow` is meaningful for corrupt only; for shuffle/filler/paraphrase it should be interpreted as a diagnostic collision with the corrupt injected value, not a planned intervention target.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--present", default=str(DEFAULT_PRESENT))
    ap.add_argument("--masked", default=str(DEFAULT_MASKED))
    ap.add_argument("--out-json", default=str(DEFAULT_JSON))
    ap.add_argument("--out-md", default=str(DEFAULT_MD))
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero unless both present/masked posthoc cells are PASS.")
    args = ap.parse_args()

    result = {
        "objective": "Posthoc answer classification for TabMWP Full-SFT full battery.",
        "cells": {
            "tabmwp_present": summarize_file(Path(args.present)),
            "tabmwp_masked": summarize_file(Path(args.masked)),
        },
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    md = render_md(result)
    out_md.write_text(md)
    print(md)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if args.strict:
        missing = [name for name, rec in result["cells"].items() if rec.get("status") != "PASS"]
        if missing:
            print("Posthoc strict check failed:", ", ".join(missing))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
