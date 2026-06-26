#!/usr/bin/env python
"""Audit Qwen3-VL-8B dense/full-SFT non-video replicas and causal probes.

This script is intentionally read-only and CPU-only. It verifies the concrete
artifacts needed for the "repeat every non-video LoRA SFT arm with 8B Full-SFT"
control, summarizes the available metrics, and makes any remaining gap explicit.
Use ``--strict`` after the final GPU runs to make missing artifacts a non-zero
exit status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POC = Path("data/distill/poc")
EXPECTED_BATTERY = {"corrupt", "shuffle", "filler", "paraphrase", "truncate", "delete"}

TRAIN_ARMS = {
    "chartqa": POC / "full_8b_chartqa/full_sft_summary.json",
    "tabmwp": POC / "full_8b_tabmwp/full_sft_summary.json",
    "finqa_b2": POC / "full_8b_finqa_b2/full_sft_summary.json",
    "finqa_vanilla": POC / "full_8b_finqa_vanilla/full_sft_summary.json",
    "finqa_b2_text": POC / "full_8b_finqa_b2_text/full_sft_summary.json",
    "finqa_vanilla_text": POC / "full_8b_finqa_vanilla_text/full_sft_summary.json",
}

BATTERY_FILES = {
    "chartqa_present": POC / "battery_full8b_chartqa_present.json",
    "chartqa_masked": POC / "battery_full8b_chartqa_masked.json",
    "tabmwp_present": POC / "battery_full8b_tabmwp_present.json",
    "tabmwp_masked": POC / "battery_full8b_tabmwp_masked.json",
    "tabmwp_present_core": POC / "battery_full8b_tabmwp_present_core.json",
    "tabmwp_masked_core": POC / "battery_full8b_tabmwp_masked_core.json",
}

FINQA_FILES = {
    "finqa_b2": POC / "battery_n1_full8b_finqa_b2.json",
    "finqa_vanilla": POC / "battery_n1_full8b_finqa_vanilla.json",
    "finqa_b2_text": POC / "battery_n1_full8b_finqa_b2_text.json",
    "finqa_vanilla_text": POC / "battery_n1_full8b_finqa_vanilla_text.json",
}

EVAL_FILES = {
    "chartqa": POC / "full_8b_chartqa/eval_n400.json",
    "tabmwp": POC / "eval_full_sft_8b_tabmwp_n400.json",
}

POSTHOC_FILE = POC / "full8b_tabmwp_battery_posthoc.json"
POSTHOC_SOURCES = {
    "tabmwp_present": BATTERY_FILES["tabmwp_present"],
    "tabmwp_masked": BATTERY_FILES["tabmwp_masked"],
}

LORA_ANALOGS = {
    "chartqa": {
        "lora_train": POC / "lora_8b_chartqa/adapter_config.json",
        "lora_eval": POC / "lora_8b_chartqa/eval_n400.json",
        "lora_probe_files": [POC / "battery_8b_present.json", POC / "battery_8b_masked.json"],
        "full_train_key": "chartqa",
        "full_eval_key": "chartqa",
        "full_probe_keys": ["chartqa_present", "chartqa_masked"],
    },
    "tabmwp": {
        "lora_train": POC / "lora_8b_tabmwp/adapter_config.json",
        "lora_eval": POC / "eval_sft_8b_tabmwp_n400.json",
        "lora_probe_files": [POC / "battery_tabmwp8b_present.json", POC / "battery_tabmwp8b_masked.json"],
        "full_train_key": "tabmwp",
        "full_eval_key": "tabmwp",
        "full_probe_keys": ["tabmwp_present", "tabmwp_masked"],
    },
    "finqa_b2": {
        "lora_train": POC / "lora_8b_finqa_b2/adapter_config.json",
        "lora_probe_file": POC / "battery_n1_8b.json",
        "lora_probe_key": "b2",
        "full_train_key": "finqa_b2",
        "full_probe_key": "finqa_b2",
    },
    "finqa_vanilla": {
        "lora_train": POC / "lora_8b_finqa_vanilla/adapter_config.json",
        "lora_probe_file": POC / "battery_n1_8b.json",
        "lora_probe_key": "vanilla",
        "full_train_key": "finqa_vanilla",
        "full_probe_key": "finqa_vanilla",
    },
    "finqa_b2_text": {
        "lora_train": POC / "lora_8b_finqa_b2_text/adapter_config.json",
        "lora_probe_file": POC / "battery_n1_8b_text.json",
        "lora_probe_key": "b2text",
        "full_train_key": "finqa_b2_text",
        "full_probe_key": "finqa_b2_text",
    },
    "finqa_vanilla_text": {
        "lora_train": POC / "lora_8b_finqa_vanilla_text/adapter_config.json",
        "lora_probe_file": POC / "battery_n1_8b_text.json",
        "lora_probe_key": "vanillatext",
        "full_train_key": "finqa_vanilla_text",
        "full_probe_key": "finqa_vanilla_text",
    },
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def battery_summary(path: Path, *, require_full: bool, require_answers: bool = False) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {
            "path": str(path), "exists": False, "complete": False,
            "missing_interventions": sorted(EXPECTED_BATTERY),
            "has_per_case_answers": False,
            "answer_variants": [],
            "missing_answer_variants": sorted(EXPECTED_BATTERY) if require_answers else [],
        }
    s = obj.get("summary", {})
    got = set((s.get("interventions") or {}).keys())
    missing = sorted(EXPECTED_BATTERY - got) if require_full else []
    rp = s.get("re_perception") or {}
    interventions = s.get("interventions") or {}
    details = obj.get("details") or []
    answer_variants = set()
    for row in details:
        answers = row.get("answers") if isinstance(row, dict) else None
        if isinstance(answers, dict):
            answer_variants.update(answers.keys())
    missing_answer_variants = sorted(got - answer_variants) if require_answers else []
    return {
        "path": str(path),
        "exists": True,
        "complete": not missing,
        "scale": s.get("scale"),
        "mask_image": s.get("mask_image"),
        "n_cases": s.get("n_cases"),
        "n_eval": s.get("n_eval"),
        "base_acc": s.get("base_acc"),
        "n_para_nums_ok": s.get("n_para_nums_ok"),
        "interventions": sorted(got),
        "missing_interventions": missing,
        "corrupt_flip": (interventions.get("corrupt") or {}).get("flip_rate"),
        "shuffle_flip": (interventions.get("shuffle") or {}).get("flip_rate"),
        "paraphrase_flip": (interventions.get("paraphrase") or {}).get("flip_rate"),
        "snap_rate": rp.get("snap_rate"),
        "follow_rate": rp.get("follow_rate"),
        "has_per_case_answers": bool(answer_variants),
        "answer_variants": sorted(answer_variants),
        "missing_answer_variants": missing_answer_variants,
    }


def finqa_summary(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {"path": str(path), "exists": False, "complete": False}
    results = obj.get("results") or {}
    if not results:
        return {"path": str(path), "exists": True, "complete": False, "error": "missing results"}
    mode, s = next(iter(results.items()))
    operand = s.get("corrupt_operand") or {}
    consistent = s.get("corrupt_consistent") or {}
    shuffle = s.get("shuffle") or {}
    return {
        "path": str(path),
        "exists": True,
        "complete": True,
        "mode": mode,
        "n_cases": s.get("n_cases"),
        "n_eval": s.get("n_eval"),
        "n_targeted": s.get("n_targeted"),
        "base_acc": s.get("base_acc"),
        "operand_follow": operand.get("follow_rate"),
        "operand_snap": operand.get("snap_rate"),
        "consistent_follow": consistent.get("follow_rate"),
        "consistent_snap": consistent.get("snap_rate"),
        "shuffle_snap": shuffle.get("snap_rate"),
    }


def train_summary(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {"path": str(path), "exists": False, "complete": False}
    return {
        "path": str(path),
        "exists": True,
        "complete": True,
        "data": obj.get("data"),
        "epochs": obj.get("epochs"),
        "train_vision": obj.get("train_vision"),
        "freeze_embeddings": obj.get("freeze_embeddings"),
        "freeze_first_layers": obj.get("freeze_first_layers"),
        "trainable_pct": obj.get("trainable_pct"),
        "n_records": obj.get("n_records"),
        "n_usable": obj.get("n_usable"),
        "train_loss": (obj.get("train_metrics") or {}).get("train_loss"),
    }


def targeted_result_exists(path: Path, key: str) -> bool:
    obj = read_json(path)
    return bool(obj and key in (obj.get("results") or {}))


def analog_coverage(train: dict[str, Any], evals: dict[str, Any],
                    batteries: dict[str, Any], finqa: dict[str, Any]) -> dict[str, dict[str, Any]]:
    coverage = {}
    for arm, cfg in LORA_ANALOGS.items():
        lora_train = Path(cfg["lora_train"]).exists()
        lora_eval = Path(cfg["lora_eval"]).exists() if cfg.get("lora_eval") else None
        if cfg.get("lora_probe_files"):
            lora_probe = all(Path(p).exists() for p in cfg["lora_probe_files"])
        else:
            lora_probe = targeted_result_exists(Path(cfg["lora_probe_file"]), cfg["lora_probe_key"])
        full_train = train[cfg["full_train_key"]]["complete"]
        full_eval = evals[cfg["full_eval_key"]]["complete"] if cfg.get("full_eval_key") else None
        if cfg.get("full_probe_keys"):
            full_probe = all(batteries[k]["complete"] for k in cfg["full_probe_keys"])
        else:
            full_probe = finqa[cfg["full_probe_key"]]["complete"]
        full_complete = full_train and full_probe and (full_eval is not False)
        source_complete = lora_train and lora_probe and (lora_eval is not False)
        coverage[arm] = {
            "source_lora_complete": source_complete,
            "source_lora_train": lora_train,
            "source_lora_eval": lora_eval,
            "source_lora_probe": lora_probe,
            "full_sft_complete": full_complete,
            "full_sft_train": full_train,
            "full_sft_eval": full_eval,
            "full_sft_probe": full_probe,
        }
    return coverage


def eval_summary(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {"path": str(path), "exists": False, "complete": False}
    return {
        "path": str(path),
        "exists": True,
        "complete": True,
        "n_eval": obj.get("n_eval"),
        "base_acc": obj.get("base_acc"),
        "best_test_acc": obj.get("best_test_acc"),
        "best_net": obj.get("best_net"),
        "best_mcnemar_p": obj.get("best_mcnemar_p"),
        "gain": ((obj.get("per_full_model") or [{}])[0]).get("gain"),
        "lost": ((obj.get("per_full_model") or [{}])[0]).get("lost"),
    }


def posthoc_summary(path: Path) -> dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {"path": str(path), "exists": False, "complete": False, "cells": {}}
    cells = obj.get("cells") or {}
    required = ["tabmwp_present", "tabmwp_masked"]
    cell_status = {}
    stale = []
    for k in required:
        rec = cells.get(k) or {}
        status = rec.get("status", "MISSING")
        expected_sha = file_sha256(POSTHOC_SOURCES[k])
        got_sha = ((rec.get("source") or {}).get("sha256"))
        if status == "PASS" and (not expected_sha or got_sha != expected_sha):
            status = "STALE"
            stale.append(k)
        cell_status[k] = status
    complete = all(cell_status.get(k) == "PASS" for k in required)
    return {
        "path": str(path),
        "exists": True,
        "complete": complete,
        "cells": cell_status,
        "stale_cells": stale,
    }


def build_audit() -> dict[str, Any]:
    train = {name: train_summary(path) for name, path in TRAIN_ARMS.items()}
    evals = {name: eval_summary(path) for name, path in EVAL_FILES.items()}
    batteries = {}
    for name, path in BATTERY_FILES.items():
        batteries[name] = battery_summary(
            path,
            require_full=not name.endswith("_core"),
            require_answers=name in {"tabmwp_present", "tabmwp_masked"},
        )
    finqa = {name: finqa_summary(path) for name, path in FINQA_FILES.items()}
    posthoc = posthoc_summary(POSTHOC_FILE)
    caches = {
        "chartqa_paraphrase": {
            "path": str(POC / "paraphrase_cache_full8b_chartqa.jsonl"),
            "lines": line_count(POC / "paraphrase_cache_full8b_chartqa.jsonl"),
        },
        "tabmwp_mimo_paraphrase": {
            "path": str(POC / "paraphrase_cache_full8b_tabmwp_mimo.jsonl"),
            "lines": line_count(POC / "paraphrase_cache_full8b_tabmwp_mimo.jsonl"),
            "expected_after_present": 387,
        },
        "tabmwp_base_cot": {
            "path": str(POC / "paraphrase_cache_full8b_tabmwp_mimo_base_cot.jsonl"),
            "lines": line_count(POC / "paraphrase_cache_full8b_tabmwp_mimo_base_cot.jsonl"),
            "expected_after_present": 400,
        },
    }
    checkpoint_weight = POC / "full_8b_tabmwp/model.safetensors"
    coverage = analog_coverage(train, evals, batteries, finqa)
    tabmwp_full_done = batteries["tabmwp_present"]["complete"] and batteries["tabmwp_masked"]["complete"]
    tabmwp_full_has_answers = (
        tabmwp_full_done
        and batteries["tabmwp_present"]["has_per_case_answers"]
        and batteries["tabmwp_masked"]["has_per_case_answers"]
        and not batteries["tabmwp_present"]["missing_answer_variants"]
        and not batteries["tabmwp_masked"]["missing_answer_variants"]
    )
    tabmwp_weight_exists = checkpoint_weight.exists()
    tabmwp_weight_retained_until_done = tabmwp_weight_exists or tabmwp_full_done
    tabmwp_weight_clean_after_done = (not tabmwp_full_done) or (not tabmwp_weight_exists)
    required = {
        "training_all_arms": all(v["complete"] for v in train.values()),
        "source_lora_8b_arms_found": all(v["source_lora_complete"] for v in coverage.values()),
        "lora_to_full_sft_8b_coverage": all(v["full_sft_complete"] for v in coverage.values()),
        "chartqa_eval": evals["chartqa"]["complete"],
        "chartqa_full_battery": batteries["chartqa_present"]["complete"] and batteries["chartqa_masked"]["complete"],
        "tabmwp_eval": evals["tabmwp"]["complete"],
        "tabmwp_full_battery": tabmwp_full_done,
        "tabmwp_full_battery_has_answers": tabmwp_full_has_answers,
        "tabmwp_posthoc_ready": posthoc["complete"],
        "finqa_targeted_all_arms": all(v["complete"] for v in finqa.values()),
        "tabmwp_weight_retained_until_battery_done": tabmwp_weight_retained_until_done,
        "tabmwp_weight_clean_after_battery_done": tabmwp_weight_clean_after_done,
    }
    missing = [name for name, ok in required.items() if not ok]
    return {
        "objective": "Qwen3-VL-8B dense/full-SFT replicas for all non-video LoRA SFT arms, with matching causal probes.",
        "complete": not missing,
        "missing_requirements": missing,
        "required": required,
        "lora_analogs": coverage,
        "training": train,
        "evals": evals,
        "tabmwp_eval": evals["tabmwp"],
        "batteries": batteries,
        "finqa": finqa,
        "posthoc": posthoc,
        "caches": caches,
        "checkpoint": {
            "tabmwp_weight_path": str(checkpoint_weight),
            "tabmwp_weight_exists": tabmwp_weight_exists,
            "tabmwp_weight_size_gb": round(checkpoint_weight.stat().st_size / (1024 ** 3), 2) if tabmwp_weight_exists else None,
            "tabmwp_full_battery_done": tabmwp_full_done,
            "policy": "retain until full battery is complete; then remove large weight shards",
        },
        "precheck_command": "PRECHECK_ONLY=1 bash scripts/resume_full_sft_8b_tabmwp_battery.sh",
        "next_gpu_command": "bash scripts/resume_full_sft_8b_tabmwp_battery.sh",
    }


def fmt_pct(x: Any) -> str:
    return "NA" if x is None else f"{100 * float(x):.1f}%"


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Full-SFT 8B Non-Video Audit",
        "",
        f"Overall: {'COMPLETE' if audit['complete'] else 'INCOMPLETE'}",
        "",
        "## Requirement Status",
        "",
        "| requirement | status |",
        "|---|---|",
    ]
    for name, ok in audit["required"].items():
        lines.append(f"| {name} | {'PASS' if ok else 'MISSING'} |")

    lines += [
        "",
        "## LoRA-to-Full-SFT 8B Coverage",
        "",
        "| arm | source LoRA train | source LoRA eval | source LoRA probe | Full-SFT train | Full-SFT eval | Full-SFT probe | Full-SFT coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in audit["lora_analogs"].items():
        def mark(v):
            return "NA" if v is None else ("PASS" if v else "MISSING")
        lines.append(
            f"| {name} | {mark(s['source_lora_train'])} | {mark(s['source_lora_eval'])} | "
            f"{mark(s['source_lora_probe'])} | {mark(s['full_sft_train'])} | "
            f"{mark(s['full_sft_eval'])} | {mark(s['full_sft_probe'])} | "
            f"{mark(s['full_sft_complete'])} |"
        )

    lines += [
        "",
        "## Training Arms",
        "",
        "| arm | status | data | usable | epochs | trainable | frozen early |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, s in audit["training"].items():
        lines.append(
            f"| {name} | {'PASS' if s['complete'] else 'MISSING'} | {s.get('data', 'NA')} | "
            f"{s.get('n_usable', 'NA')} | {s.get('epochs', 'NA')} | {fmt_pct(s.get('trainable_pct'))} | "
            f"{s.get('freeze_first_layers', 'NA')} |"
        )

    lines += [
        "",
        "## Full-SFT Eval",
        "",
        "| dataset | status | n_eval | base_acc | full_acc | net | gain/lost | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in audit["evals"].items():
        lines.append(
            f"| {name} | {'PASS' if s['complete'] else 'MISSING'} | {s.get('n_eval', 'NA')} | "
            f"{fmt_pct(s.get('base_acc'))} | {fmt_pct(s.get('best_test_acc'))} | "
            f"{fmt_pct(s.get('best_net'))} | {s.get('gain', 'NA')}/{s.get('lost', 'NA')} | "
            f"{s.get('best_mcnemar_p', 'NA')} |"
        )

    lines += [
        "",
        "## Chart/Table Batteries",
        "",
        "| cell | status | n_eval | base_acc | interventions | per-case answers | answer variants | missing answer variants | corrupt_flip | shuffle_flip | para_flip | snap | follow |",
        "|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in audit["batteries"].items():
        status = "PASS" if s["complete"] else ("PARTIAL" if s["exists"] else "MISSING")
        interventions = ",".join(s.get("interventions") or [])
        lines.append(
            f"| {name} | {status} | {s.get('n_eval', 'NA')} | {fmt_pct(s.get('base_acc'))} | "
            f"{interventions or 'NA'} | {'YES' if s.get('has_per_case_answers') else 'NO'} | "
            f"{','.join(s.get('answer_variants') or []) or 'NA'} | "
            f"{','.join(s.get('missing_answer_variants') or []) or 'NA'} | "
            f"{fmt_pct(s.get('corrupt_flip'))} | {fmt_pct(s.get('shuffle_flip'))} | "
            f"{fmt_pct(s.get('paraphrase_flip'))} | {fmt_pct(s.get('snap_rate'))} | {fmt_pct(s.get('follow_rate'))} |"
        )

    lines += [
        "",
        "## FinQA Targeted",
        "",
        "| arm | status | n_eval | n_targeted | base_acc | operand_follow | consistent_follow | shuffle_snap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in audit["finqa"].items():
        lines.append(
            f"| {name} | {'PASS' if s['complete'] else 'MISSING'} | {s.get('n_eval', 'NA')} | "
            f"{s.get('n_targeted', 'NA')} | {fmt_pct(s.get('base_acc'))} | "
            f"{fmt_pct(s.get('operand_follow'))} | {fmt_pct(s.get('consistent_follow'))} | "
            f"{fmt_pct(s.get('shuffle_snap'))} |"
        )

    te = audit["evals"]["tabmwp"]
    lines += [
        "",
        "## TabMWP Eval",
        "",
        f"- base -> full: {fmt_pct(te.get('base_acc'))} -> {fmt_pct(te.get('best_test_acc'))}",
        f"- net: {fmt_pct(te.get('best_net'))}; gain/lost: {te.get('gain', 'NA')}/{te.get('lost', 'NA')}; McNemar p: {te.get('best_mcnemar_p', 'NA')}",
        "",
        "## TabMWP Posthoc",
        "",
        f"- status: {'PASS' if audit['posthoc'].get('complete') else 'MISSING'}",
        f"- path: `{audit['posthoc'].get('path')}`",
        f"- cells: {audit['posthoc'].get('cells', {})}",
        "",
        "## Resume Readiness",
        "",
        "| item | current | expected/notes |",
        "|---|---:|---|",
    ]
    caches = audit.get("caches") or {}
    for key in ["tabmwp_mimo_paraphrase", "tabmwp_base_cot"]:
        c = caches.get(key) or {}
        current = c.get("lines")
        lines.append(
            f"| {key} | {current if current is not None else 'MISSING'} | "
            f"{c.get('expected_after_present', 'NA')} after present run; {c.get('path', 'NA')} |"
        )
    ckpt = audit.get("checkpoint") or {}
    lines += [
        f"| tabmwp_weight | {'present' if ckpt.get('tabmwp_weight_exists') else 'absent'} | "
        f"{ckpt.get('tabmwp_weight_size_gb', 'NA')} GB; {ckpt.get('policy', 'NA')} |",
        "",
        "## Remaining Work",
        "",
    ]
    if audit["complete"]:
        lines.append("- None.")
    else:
        for item in audit["missing_requirements"]:
            lines.append(f"- {item}")
        lines.append(f"- Precheck command: `{audit['precheck_command']}`")
        lines.append(f"- Resume GPU command: `{audit['next_gpu_command']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if the full objective is incomplete.")
    args = ap.parse_args()

    audit = build_audit()
    md = render_markdown(audit)
    print(md)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md)
    if args.strict and not audit["complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
