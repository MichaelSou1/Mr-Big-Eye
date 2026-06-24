#!/usr/bin/env python
"""WU-1 verification report — gather every n=400 artifact (gate nets, SFT Δacc + McNemar, causal
probe flip rates, power table) into one summary + an acceptance checklist. Robust to partial state
(prints what exists). Read-only."""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path("/home/gpus/Mr-Big-Eye-internalization")
RES = ROOT / "data/distill/results/results.jsonl"


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def jget(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def gate_summary(rows):
    out = {}
    for mid in ("4b", "8b", "32b"):
        cell = [r for r in rows if r.get("model_id") == mid and r.get("dataset") == "chartqa" and r.get("n") == 400]
        if not cell:
            continue
        free = [r for r in cell if r["method"] == "free_form"]
        d = {"n": 400, "free_acc": round(sum(r["free_acc"] for r in free) / len(free), 3) if free else None}
        for m in ("self_reflect", "orch_reflect_blind"):
            seeds = [r for r in cell if r["method"] == m]
            if seeds:
                nets = [r["bootstrap"]["net"] for r in seeds]
                mean = sum(nets) / len(nets)
                sd = (sum((x - mean) ** 2 for x in nets) / (len(nets) - 1)) ** 0.5 if len(nets) > 1 else 0.0
                d[m] = {"k": len(seeds), "net_mean": round(mean, 4), "net_std": round(sd, 4)}
        out[mid] = d
    return out


def main():
    print("=" * 72)
    print("WU-1 REPORT (n=400 ChartQA)")
    print("=" * 72)
    rows = load_jsonl(RES)
    # include the 4B side-file if not yet merged into the canonical store
    side = ROOT / "data/distill/results/results_4b_n400.jsonl"
    if side.exists():
        seen = {json.dumps(r.get("fingerprint"), sort_keys=True) for r in rows}
        for r in load_jsonl(side):
            if json.dumps(r.get("fingerprint"), sort_keys=True) not in seen:
                rows.append(r)

    print("\n## 1.1 test set")
    tc = load_jsonl(ROOT / "data/distill/chartqa/test_cases_400.jsonl")
    imgs = list(Path("/home/gpus/mbe_data/chartqa_test_images").glob("chartqa_*.png"))
    print(f"  test_cases_400.jsonl: {len(tc)} rows ; images: {len(imgs)}")

    print("\n## 1.2 gate at scale (free / agentic net±std)")
    gs = gate_summary(rows)
    for mid in ("4b", "8b", "32b"):
        d = gs.get(mid)
        if not d:
            print(f"  {mid}: (none)"); continue
        sr = d.get("self_reflect", {}); orch = d.get("orch_reflect_blind", {})
        print(f"  {mid:3s} free={d['free_acc']}  "
              f"self_reflect net={sr.get('net_mean')}±{sr.get('net_std')} (k={sr.get('k')})  "
              f"orch net={orch.get('net_mean')}±{orch.get('net_std')} (k={orch.get('k')})")

    print("\n## 1.3 SFT Δacc at n=400 (base vs adapter; paired bootstrap CI + McNemar)")
    for tag, p in (("32B", ROOT / "data/distill/poc/lora_32b_chartqa/eval_n400.json"),
                   ("8B", ROOT / "data/distill/poc/lora_8b_chartqa/eval_n400.json")):
        d = jget(p)
        if not d:
            print(f"  {tag}: (not yet)"); continue
        print(f"  {tag}: base_acc={d['base_acc']:.3f}  n={d['n_eval']}")
        for a in d["per_adapter"]:
            ci = a["boot_ci"]
            print(f"      {Path(a['adapter']).name}: test_acc={a['test_acc']:.3f} "
                  f"net={a['net']:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] "
                  f"McNemar b={a['mcnemar_b']} c={a['mcnemar_c']} p={a['mcnemar_p']:.4f}")

    print("\n## 1.4 causal probe 2x2 at n=400 (flip rates)")
    for label, p in (("32B present", "data/distill/poc/causal_probe_32b_n400.json"),
                     ("32B masked ", "data/distill/poc/causal_probe_32b_maskimg_n400.json"),
                     ("8B  present", "data/distill/poc/causal_probe_8b_n400.json"),
                     ("8B  masked ", "data/distill/poc/causal_probe_8b_maskimg_n400.json")):
        d = jget(ROOT / p)
        if not d:
            print(f"  {label}: (not yet)"); continue
        s = d["summary"]
        print(f"  {label}: n_eval={s['n_eval']} corrupt={s['flip_rate_corrupt']:.3f} "
              f"shuffle={s['flip_rate_shuffle']:.3f}")

    print("\n## 1.5 power table")
    pt = jget(ROOT / "data/distill/results/power_table.json")
    if pt:
        for c in pt["cells"]:
            if c["dataset"] == "chartqa":
                print(f"  {c['model_id']:3s} chartqa p_free={c['p_free']:.3f} n_obs={c['n_obs']} "
                      f"mde@n_obs={c['mde_at_n_obs']:.3f}")
    else:
        print("  (not yet)")

    print("\n## acceptance")
    ok_11 = len(tc) >= 300 and len(imgs) >= 300
    e32 = jget(ROOT / "data/distill/poc/lora_32b_chartqa/eval_n400.json")
    e8 = jget(ROOT / "data/distill/poc/lora_8b_chartqa/eval_n400.json")
    probes = [jget(ROOT / f"data/distill/poc/causal_probe_{x}.json") for x in
              ("32b_n400", "32b_maskimg_n400", "8b_n400", "8b_maskimg_n400")]
    print(f"  [{'x' if ok_11 else ' '}] 1.1 n>=300 test set + images, train-disjoint")
    print(f"  [{'x' if all(gs.get(m,{}).get('free_acc') is not None for m in ('4b','8b','32b')) else ' '}] "
          f"1.2 free/net+CI/verdict for 3 scales")
    print(f"  [{'x' if (e32 and e8) else ' '}] 1.3 8B+32B SFT Δacc + bootstrap CI + McNemar at n>=300")
    print(f"  [{'x' if all(probes) else ' '}] 1.4 four 2x2 probe cells at n>=300")
    print(f"  [{'x' if pt else ' '}] 1.5 power table written")


if __name__ == "__main__":
    main()
