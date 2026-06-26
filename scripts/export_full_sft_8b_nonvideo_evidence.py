#!/usr/bin/env python
"""Export paper-facing evidence tables and a TabMWP resume manifest.

Read-only with respect to experiments: this script consumes the existing audit
and result JSON/JSONL files, then writes lightweight Markdown/JSON summaries.
It does not load a model or use the GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_full_sft_8b_nonvideo import build_audit


POC = Path("data/distill/poc")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    for line in path.open():
        if line.strip():
            e = json.loads(line)
            out[e["key"]] = e
    return out


def pct(x: Any, digits: int = 1) -> str:
    if x is None:
        return "NA"
    return f"{100 * float(x):.{digits}f}%"


def ratio(x: Any) -> str:
    return "NA" if x is None else str(x)


def battery_metrics(audit: dict[str, Any], name: str) -> dict[str, Any]:
    s = audit["batteries"].get(name) or {}
    corrupt = s.get("corrupt_flip")
    shuffle = s.get("shuffle_flip")
    return {
        "status": "PASS" if s.get("complete") else ("PARTIAL" if s.get("exists") else "PENDING"),
        "n_eval": s.get("n_eval"),
        "base_acc": s.get("base_acc"),
        "corrupt_flip": corrupt,
        "shuffle_flip": shuffle,
        "faithfulness_f": None if corrupt is None or shuffle is None else corrupt - shuffle,
        "paraphrase_flip": s.get("paraphrase_flip"),
        "snap_rate": s.get("snap_rate"),
        "follow_rate": s.get("follow_rate"),
        "interventions": s.get("interventions") or [],
    }


def finqa_metrics(audit: dict[str, Any], name: str) -> dict[str, Any]:
    s = audit["finqa"].get(name) or {}
    return {
        "status": "PASS" if s.get("complete") else "PENDING",
        "n_eval": s.get("n_eval"),
        "n_targeted": s.get("n_targeted"),
        "base_acc": s.get("base_acc"),
        "operand_follow": s.get("operand_follow"),
        "operand_snap": s.get("operand_snap"),
        "consistent_follow": s.get("consistent_follow"),
        "consistent_snap": s.get("consistent_snap"),
        "shuffle_snap": s.get("shuffle_snap"),
    }


def build_control_table(audit: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-VL-8B Dense/Full-SFT Non-Video Control Evidence",
        "",
        "Dense/full-SFT here means the vision tower is frozen, embeddings are frozen,",
        "and the first 3 language layers are frozen; ChartQA/TabMWP summaries report",
        "about 79.7% trainable parameters.",
        "",
        "## LoRA-to-Full-SFT 8B Coverage",
        "",
        "| arm | source LoRA train | source LoRA eval | source LoRA probe | Full-SFT train | Full-SFT eval | Full-SFT probe | Full-SFT coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in (audit.get("lora_analogs") or {}).items():
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
        "## SFT Accuracy Control",
        "",
        "| dataset/arm | base acc | full-SFT acc | net | gain/lost | McNemar p | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ["chartqa", "tabmwp"]:
        ev = (audit.get("evals") or {}).get(name) or (audit.get("tabmwp_eval") if name == "tabmwp" else {})
        status = "PASS" if ev.get("complete") else "PENDING"
        lines.append(
            f"| {name} | {pct(ev.get('base_acc'))} | {pct(ev.get('best_test_acc'))} | "
            f"{pct(ev.get('best_net'))} | {ev.get('gain', 'NA')}/{ev.get('lost', 'NA')} | "
            f"{ev.get('best_mcnemar_p', 'NA')} | {status} |"
        )

    lines += [
        "",
        "## Chart/Table Causal Battery",
        "",
        "| cell | status | n_eval | base/free acc | corrupt flip | shuffle flip | F=corrupt-shuffle | paraphrase flip | snap | follow | interventions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ["chartqa_present", "chartqa_masked", "tabmwp_present_core", "tabmwp_masked_core", "tabmwp_present", "tabmwp_masked"]:
        m = battery_metrics(audit, name)
        lines.append(
            f"| {name} | {m['status']} | {ratio(m['n_eval'])} | {pct(m['base_acc'])} | "
            f"{pct(m['corrupt_flip'])} | {pct(m['shuffle_flip'])} | {pct(m['faithfulness_f'])} | "
            f"{pct(m['paraphrase_flip'])} | {pct(m['snap_rate'])} | {pct(m['follow_rate'])} | "
            f"{', '.join(m['interventions']) or 'NA'} |"
        )

    lines += [
        "",
        "## FinQA Targeted Causal Probe",
        "",
        "| arm | status | n_eval | n_targeted | base acc | operand follow | operand snap | consistent follow | consistent snap | shuffle snap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["finqa_b2", "finqa_vanilla", "finqa_b2_text", "finqa_vanilla_text"]:
        m = finqa_metrics(audit, name)
        lines.append(
            f"| {name} | {m['status']} | {ratio(m['n_eval'])} | {ratio(m['n_targeted'])} | "
            f"{pct(m['base_acc'])} | {pct(m['operand_follow'])} | {pct(m['operand_snap'])} | "
            f"{pct(m['consistent_follow'])} | {pct(m['consistent_snap'])} | {pct(m['shuffle_snap'])} |"
        )

    posthoc = audit.get("posthoc") or {}
    lines += [
        "",
        "## TabMWP Posthoc Answer Classification",
        "",
        f"- status: {'PASS' if posthoc.get('complete') else 'PENDING'}",
        f"- path: `{posthoc.get('path', 'data/distill/poc/full8b_tabmwp_battery_posthoc.json')}`",
        f"- cells: {posthoc.get('cells', {})}",
        "",
    ]

    lines += [
        "",
        "## Current Interpretation",
        "",
        "- ChartQA full battery and FinQA targeted probes already show the same qualitative signature as the LoRA controls.",
        "- TabMWP has strong dense/full-SFT accuracy and core corrupt/shuffle evidence, but the full six-intervention present/masked battery is still pending.",
        "- Do not claim the non-video Full-SFT control is complete until `scripts/audit_full_sft_8b_nonvideo.py --strict` passes.",
        "",
    ]
    return "\n".join(lines)


def build_resume_manifest(audit: dict[str, Any]) -> dict[str, Any]:
    old_cache = load_jsonl_map(POC / "paraphrase_cache_full8b_tabmwp.jsonl")
    mimo_cache = load_jsonl_map(POC / "paraphrase_cache_full8b_tabmwp_mimo.jsonl")
    base_cot_cache = load_jsonl_map(POC / "paraphrase_cache_full8b_tabmwp_mimo_base_cot.jsonl")
    missing = sorted(set(old_cache) - set(mimo_cache), key=lambda k: int(k.rsplit("-", 1)[-1]))
    mismatch = sorted(k for k in set(old_cache) & set(mimo_cache)
                      if old_cache[k].get("base_md5") != mimo_cache[k].get("base_md5"))
    checkpoint = audit["checkpoint"]
    return {
        "objective": "Resume only the remaining TabMWP Qwen3-VL-8B dense/full-SFT six-intervention batteries.",
        "cpu_readiness_precheck_command": "PRECHECK_ONLY=1 REQUIRE_GPU_IDLE=0 bash scripts/resume_full_sft_8b_tabmwp_battery.sh",
        "precheck_command": "PRECHECK_ONLY=1 bash scripts/resume_full_sft_8b_tabmwp_battery.sh",
        "gpu_command": "bash scripts/resume_full_sft_8b_tabmwp_battery.sh",
        "finalize_command": "bash scripts/finalize_full_sft_8b_nonvideo.sh",
        "strict_audit_command": "/home/gpus/anaconda3/envs/mbe-up/bin/python scripts/audit_full_sft_8b_nonvideo.py --strict",
        "posthoc_command": "/home/gpus/anaconda3/envs/mbe-up/bin/python scripts/summarize_full8b_tabmwp_posthoc.py",
        "posthoc_strict_command": "/home/gpus/anaconda3/envs/mbe-up/bin/python scripts/summarize_full8b_tabmwp_posthoc.py --strict",
        "resource_guards": {
            "require_gpu_idle_default": True,
            "max_gpu_used_mb_default": 2048,
            "min_gpu_free_mb_default": 16000,
            "min_disk_free_gb_default": 40,
            "require_orchestrator_default": True,
            "orchestrator_host_hint_default": "xiaomimimo.com",
            "expected_interventions_default": "corrupt,delete,filler,paraphrase,shuffle,truncate",
            "override": "REQUIRE_GPU_IDLE=0 only when intentionally sharing/using non-idle GPUs.",
        },
        "expected_outputs": [
            "data/distill/poc/battery_full8b_tabmwp_present.json",
            "data/distill/poc/battery_full8b_tabmwp_masked.json",
        ],
        "existing_core_outputs": [
            "data/distill/poc/battery_full8b_tabmwp_present_core.json",
            "data/distill/poc/battery_full8b_tabmwp_masked_core.json",
        ],
        "posthoc_outputs": [
            "data/distill/poc/full8b_tabmwp_battery_posthoc.json",
            "docs/reviews/full8b_tabmwp_battery_posthoc.md",
        ],
        "cache_state": {
            "old_cache": "data/distill/poc/paraphrase_cache_full8b_tabmwp.jsonl",
            "old_cache_entries": len(old_cache),
            "mimo_cache": "data/distill/poc/paraphrase_cache_full8b_tabmwp_mimo.jsonl",
            "mimo_cache_entries": len(mimo_cache),
            "mimo_missing_count": len(missing),
            "mimo_missing_keys": missing,
            "base_cot_cache": "data/distill/poc/paraphrase_cache_full8b_tabmwp_mimo_base_cot.jsonl",
            "base_cot_cache_entries": len(base_cot_cache),
            "base_cot_expected_after_present": 400,
            "common_base_md5_mismatches": mismatch,
        },
        "checkpoint": checkpoint,
        "audit_missing_requirements": audit["missing_requirements"],
        "notes": [
            "While the user's GPUs are reserved, use cpu_readiness_precheck_command only; it skips the GPU idle query and exits before model load.",
            "Do not delete the TabMWP weight shard before both expected_outputs exist.",
            "Existing output JSONs are skipped only if they contain all six interventions and details[].answers; incomplete outputs are moved to *.incomplete.<timestamp> before rerun.",
            "After both expected_outputs exist, the resume script removes large TabMWP weight shards unless KEEP_CHECKPOINTS=1.",
            "The strict audit requires the large TabMWP weight shard to be gone after the full battery is complete.",
            "The old non-Mimo cache is complete and base_md5-compatible for common entries, but the resume path intentionally uses the Mimo cache.",
            "The base-CoT cache is generated by the local Full-SFT model, so missing base-CoT entries cannot be filled without a GPU run.",
            "Base-CoT cache hits are bound to a lightweight checkpoint fingerprint; stale or old-format cache entries are regenerated instead of reused.",
            "Run the precheck command first; it exits before model load and enforces the GPU/disk guards.",
            "The finalize command also runs the posthoc command to classify shuffle/filler/paraphrase answers from details[].answers.",
            "The next GPU run will generate/base-cache CoTs as needed, then finish remaining Mimo paraphrases and continuation probes.",
        ],
    }


def manifest_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# TabMWP Full-SFT Battery Resume Manifest",
        "",
        f"- CPU-only readiness precheck: `{manifest['cpu_readiness_precheck_command']}`",
        f"- Precheck command: `{manifest['precheck_command']}`",
        f"- GPU command: `{manifest['gpu_command']}`",
        f"- Finalize command: `{manifest['finalize_command']}`",
        f"- Strict audit: `{manifest['strict_audit_command']}`",
        f"- Posthoc command: `{manifest['posthoc_command']}`",
        f"- Posthoc strict check: `{manifest['posthoc_strict_command']}`",
        f"- Resource guard: GPU idle required by default; max used {manifest['resource_guards']['max_gpu_used_mb_default']}MB/GPU, min free {manifest['resource_guards']['min_gpu_free_mb_default']}MB/GPU, min disk {manifest['resource_guards']['min_disk_free_gb_default']}GB; orchestrator host must include `{manifest['resource_guards']['orchestrator_host_hint_default']}`",
        f"- Required interventions for completed outputs: `{manifest['resource_guards']['expected_interventions_default']}` plus `details[].answers`",
        f"- Missing Mimo paraphrases: {manifest['cache_state']['mimo_missing_count']}",
        f"- Mimo cache entries: {manifest['cache_state']['mimo_cache_entries']}",
        f"- Old cache entries: {manifest['cache_state']['old_cache_entries']}",
        f"- Base-CoT cache entries: {manifest['cache_state']['base_cot_cache_entries']}/{manifest['cache_state']['base_cot_expected_after_present']}",
        f"- Common base_md5 mismatches: {len(manifest['cache_state']['common_base_md5_mismatches'])}",
        f"- TabMWP weight exists: {manifest['checkpoint']['tabmwp_weight_exists']} ({manifest['checkpoint']['tabmwp_weight_size_gb']} GB)",
        "",
        "## Expected Outputs",
        "",
    ]
    for p in manifest["expected_outputs"]:
        lines.append(f"- `{p}`")
    lines += ["", "## Posthoc Outputs", ""]
    for p in manifest["posthoc_outputs"]:
        lines.append(f"- `{p}`")
    lines += ["", "## Missing Mimo Keys", ""]
    for key in manifest["cache_state"]["mimo_missing_keys"]:
        lines.append(f"- `{key}`")
    lines += ["", "## Notes", ""]
    for note in manifest["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-md", default="docs/reviews/full_sft_8b_nonvideo_control_table.md")
    ap.add_argument("--manifest-json", default="data/distill/poc/full_sft_8b_tabmwp_resume_manifest.json")
    ap.add_argument("--manifest-md", default="docs/reviews/full_sft_8b_tabmwp_resume_manifest.md")
    args = ap.parse_args()

    audit = build_audit()
    table_md = build_control_table(audit)
    manifest = build_resume_manifest(audit)
    manifest_md = manifest_markdown(manifest)

    for path, text in [
        (Path(args.table_md), table_md),
        (Path(args.manifest_md), manifest_md),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    mpath = Path(args.manifest_json)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(table_md)
    print(f"Wrote {args.table_md}")
    print(f"Wrote {args.manifest_json}")
    print(f"Wrote {args.manifest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
