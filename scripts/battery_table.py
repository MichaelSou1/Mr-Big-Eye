#!/usr/bin/env python
"""WU-2 — aggregate battery runs into the acceptance artifacts (no hand-copied numbers).

Reads the four per-run JSONs produced by ``battery_n400.py``
({8b,32b} x {present,masked}) and emits:
  - battery table: intervention x scale x {present,masked} -> flip-rate AND accuracy-after, n_eval
  - F = flip_corrupt - flip_shuffle per cell (continuity with WU-1.4)
  - re-perception snap-rate (present, per scale)
  - early-answering curve (truncate frac -> flip, acc)
  - delete-vs-k curve (k -> flip, acc)
  - control pairs: paraphrase (corrupt's two-sided control), filler (shuffle's control)
to data/distill/poc/battery_table.json and prints a Markdown summary.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = {
    ("8b", "present"): "data/distill/poc/battery_8b_present.json",
    ("8b", "masked"): "data/distill/poc/battery_8b_masked.json",
    ("32b", "present"): "data/distill/poc/battery_32b_present.json",
    ("32b", "masked"): "data/distill/poc/battery_32b_masked.json",
}
OUT = ROOT / "data/distill/poc/battery_table.json"


def load_runs():
    runs = {}
    for key, rel in RUN_FILES.items():
        p = ROOT / rel
        if p.exists():
            runs[key] = json.loads(p.read_text())["summary"]
        else:
            print(f"WARN missing run: {rel}")
    return runs


def fr(m):  # flip_rate / acc cell
    return f"{m['flip_rate']:.3f}/{m['acc_after']:.3f}" if m else "—"


def main() -> int:
    runs = load_runs()
    if not runs:
        print("no runs found; run battery_n400.py first."); return 1

    cells = list(RUN_FILES.keys())
    lines = []
    lines.append("# WU-2 faithfulness battery — flip_rate / accuracy_after (n_eval)\n")
    lines.append("Each cell = answer flip-rate (vs model's base answer) / accuracy-after (vs gold).\n")
    hdr = "| intervention | " + " | ".join(f"{s} {c}" for s, c in cells) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(cells) + 1))

    def row(label, getter):
        vals = []
        for cell in cells:
            s = runs.get(cell)
            vals.append(fr(getter(s)) if s else "—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    iv = lambda s, name: (s.get("interventions", {}) or {}).get(name) if s else None
    row("corrupt", lambda s: iv(s, "corrupt"))
    row("shuffle", lambda s: iv(s, "shuffle"))
    row("paraphrase (corrupt-ctrl)", lambda s: iv(s, "paraphrase"))
    row("filler (shuffle-ctrl)", lambda s: iv(s, "filler"))
    for f in ["0.25", "0.5", "0.75"]:
        row(f"truncate@{f}", lambda s, f=f: (iv(s, "truncate") or {}).get(f))
    for k in ["1", "2", "3"]:
        row(f"delete@k={k}", lambda s, k=k: (iv(s, "delete") or {}).get(k))

    # F = flip_corrupt - flip_shuffle
    lines.append("\n## F = flip_corrupt − flip_shuffle (≤0 ⇒ corrupt no worse than shuffle ⇒ CoT not load-bearing)\n")
    lines.append("| cell | n_eval | corrupt | shuffle | F |")
    lines.append("|---|---|---|---|---|")
    for cell in cells:
        s = runs.get(cell)
        if not s:
            continue
        c = iv(s, "corrupt"); sh = iv(s, "shuffle")
        if c and sh:
            F = c["flip_rate"] - sh["flip_rate"]
            lines.append(f"| {cell[0]} {cell[1]} | {s['n_eval']} | {c['flip_rate']:.3f} | "
                         f"{sh['flip_rate']:.3f} | {F:+.3f} |")

    # re-perception (present)
    lines.append("\n## N2 re-perception (present): after corrupting an intermediate, does the answer "
                 "snap to the TRUE value (re-read) or follow the injected wrong value (load-bearing)?\n")
    lines.append("| scale | n_corrupt | snap_to_true | follows_injected | other | **snap_rate** | follow_rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for s_name in ["8b", "32b"]:
        s = runs.get((s_name, "present"))
        rp = s.get("re_perception") if s else None
        if rp:
            lines.append(f"| {s_name} | {rp['n_corrupt']} | {rp['snap_to_true']} | "
                         f"{rp['follows_injected']} | {rp['other']} | **{rp['snap_rate']:.3f}** | "
                         f"{rp['follow_rate']:.3f} |")

    # early-answering curve
    lines.append("\n## Early-answering curve (truncate frac → flip_rate, present)\n")
    lines.append("| scale | @0.25 | @0.5 | @0.75 |")
    lines.append("|---|---|---|---|")
    for s_name in ["8b", "32b"]:
        s = runs.get((s_name, "present"))
        tr = iv(s, "truncate") if s else None
        if tr:
            lines.append(f"| {s_name} | " + " | ".join(
                f"{tr[f]['flip_rate']:.3f}" if tr.get(f) else "—" for f in ["0.25", "0.5", "0.75"]) + " |")

    # paraphrase number-fidelity
    lines.append("\n## paraphrase number-fidelity (DeepSeek, multiset preserved)\n")
    for cell in cells:
        s = runs.get(cell)
        if s and cell[1] == "present":
            lines.append(f"- {cell[0]}: {s.get('n_para_nums_ok','?')}/{s['n_eval']} CoTs numbers preserved")

    md = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({k[0] + "_" + k[1]: v for k, v in runs.items()},
                              ensure_ascii=False, indent=2))
    print(md)
    (ROOT / "data/distill/poc/battery_table.md").write_text(md)
    print(f"\nwrote {OUT} and battery_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
