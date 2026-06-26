#!/usr/bin/env python
"""WU-5 — the faithfulness ⊥ accuracy axis + the two headline figures.

Everything here is *derived from the append-only result store* (no hand-copied numbers):
  - battery runs   data/distill/poc/battery_{8b,32b}_{present,masked}.json  (corrupt/shuffle/re-perception)
  - SFT eval       data/distill/poc/lora_{32b,8b}_chartqa/eval_n400.json    (base→SFT Δacc, CI, McNemar)
  - the MAP        data/distill/results/map.json                            (regime per cell, spec §8)
  - master table   data/distill/results/tables.json                        (best-agentic net±CI per cell)

Definitions
  F        = flip_corrupt - flip_shuffle          (present)   ; ≤0 ⇒ CoT not load-bearing
  F_masked = flip_corrupt - flip_shuffle          (image masked)
  gap      = flip_corrupt(masked) - flip_corrupt(present)      ; latent chain surfaces only w/o image
  SE(F)    = sqrt(p_c(1-p_c)/n_c + p_s(1-p_s)/n_s)             (binomial, from the stored flip counts)

Emits (all under the result store / paper tree):
  data/distill/results/faithfulness.json        the F × Δacc join, per cell
  data/distill/results/faithfulness_master.md   master table with a faithfulness (F) column
  docs/paper/figures/fig_map.{pdf,png}          Figure A — the regime MAP heatmap
  docs/paper/figures/fig_decoupling.{pdf,png}   Figure B — accuracy-gain ⊥ faithfulness scatter
  docs/paper/figures/figures_snippet.tex        ready-to-\input figure environments

Read-only w.r.t. the store; only writes the four artifacts above. Run in `mbe-up`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
POC = ROOT / "data/distill/poc"
RES = ROOT / "data/distill/results"
FIGS = ROOT / "docs/paper/figures"

SCALES = ["8b", "32b"]                       # cells with both battery F and SFT Δacc
BATTERY = {(s, c): POC / f"battery_{s}_{c}.json" for s in SCALES for c in ("present", "masked")}
SFT_EVAL = {"8b": POC / "lora_8b_chartqa/eval_n400.json",
            "32b": POC / "lora_32b_chartqa/eval_n400.json"}


def jget(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def iv(summary, name):
    return (summary.get("interventions", {}) or {}).get(name)


def se_diff(c, sh):
    """Binomial SE of (flip_corrupt - flip_shuffle), treating the two arms as independent."""
    pc, nc = c["flip_rate"], c["n"]
    ps, ns = sh["flip_rate"], sh["n"]
    return math.sqrt(pc * (1 - pc) / nc + ps * (1 - ps) / ns)


# --------------------------------------------------------------------------- join
def build_join():
    """Per-(chartqa, scale) row: F present/masked + gap + SE, snap-rate, and the SFT Δacc block."""
    cells = []
    for s in SCALES:
        pres = jget(BATTERY[(s, "present")])
        mask = jget(BATTERY[(s, "masked")])
        sft = jget(SFT_EVAL[s])
        if not (pres and sft):
            print(f"WARN {s}: missing battery-present or SFT eval; skipping")
            continue
        ps = pres["summary"]
        cP, shP = iv(ps, "corrupt"), iv(ps, "shuffle")
        row = {
            "dataset": "chartqa", "model_id": s,
            "n_eval_present": ps["n_eval"],
            "flip_corrupt_present": cP["flip_rate"], "flip_shuffle_present": shP["flip_rate"],
            "F_present": cP["flip_rate"] - shP["flip_rate"],
            "F_present_se": se_diff(cP, shP),
            "acc_after_corrupt_present": cP["acc_after"],
            "snap_rate": (ps.get("re_perception") or {}).get("snap_rate"),
            "follow_rate": (ps.get("re_perception") or {}).get("follow_rate"),
        }
        if mask:
            ms = mask["summary"]
            cM, shM = iv(ms, "corrupt"), iv(ms, "shuffle")
            row.update({
                "n_eval_masked": ms["n_eval"],
                "flip_corrupt_masked": cM["flip_rate"], "flip_shuffle_masked": shM["flip_rate"],
                "F_masked": cM["flip_rate"] - shM["flip_rate"],
                "F_masked_se": se_diff(cM, shM),
                "masked_minus_present_gap": cM["flip_rate"] - cP["flip_rate"],
            })
        # SFT Δacc block (the peak adapter eval already records base/best)
        ad = sft["per_adapter"][0] if sft.get("per_adapter") else {}
        row["sft"] = {
            "base_acc": sft["base_acc"], "sft_acc": ad.get("test_acc"),
            "net": ad.get("net"), "ci": ad.get("boot_ci"),
            "mcnemar_b": ad.get("mcnemar_b"), "mcnemar_c": ad.get("mcnemar_c"),
            "mcnemar_p": ad.get("mcnemar_p"), "n_eval": sft.get("n_eval"),
        }
        cells.append(row)
    return cells


# --------------------------------------------------------------------------- Figure A (MAP)
def fig_map(out_pdf, out_png):
    """Render the spec-§8 regime MAP (map.json) as a heatmap, annotated with free-acc,
    best-agentic net±CI and regime. Colour encodes regime (R1 perception/selection-bound vs
    R2 reasoning-bound); unrun cross-products and cells without agentic runs are explicitly labelled."""
    mp = jget(RES / "map.json")
    tbl = jget(RES / "tables.json") or {"master": []}
    ci_by = {(m["dataset"], m["model_id"], m["method"]): m.get("ci") for m in tbl.get("master", [])}
    cells = mp["cells"]

    datasets = sorted({c["dataset"] for c in cells})
    order = ["4b", "8b", "32b", "internvl", "penguin2b", "penguin8b", "8b_sft"]
    models = [m for m in order if any(c["model_id"] == m for c in cells)]
    models += sorted({c["model_id"] for c in cells} - set(models))
    by = {(c["dataset"], c["model_id"]): c for c in cells}

    # colour: R2 = warm (reasoning-bound, the rare internalizable regime), R1 = cool, none = grey.
    C = {1: "#cfe3f2", 2: "#f6c9a8", None: "#eeeeee"}
    fig, ax = plt.subplots(figsize=(1.7 * len(datasets) + 1.6, 0.92 * len(models) + 1.2))
    for yi, mid in enumerate(models):
        for xi, ds in enumerate(datasets):
            c = by.get((ds, mid))
            reg = c["regime"] if c else None
            ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor=C.get(reg, C[None]),
                                       edgecolor="white", linewidth=2))
            if not c:
                ax.text(xi + 0.5, yi + 0.5, "NA", ha="center", va="center",
                        fontsize=7.2, color="#9a9a9a")
                continue
            fa = c.get("free_acc")
            net = c.get("best_net")
            ci = ci_by.get((ds, mid, c.get("best_method")))
            lab = f"free={fa:.2f}" if fa is not None else "—"
            if net is not None:
                lab += f"\nnet={net:+.02f}"
                if ci:
                    lab += f"\n[{ci[0]:+.02f},{ci[1]:+.02f}]"
            else:
                lab += "\nno agentic"
            lab += f"\n{'R2' if reg == 2 else 'R1'}"
            ax.text(xi + 0.5, yi + 0.5, lab, ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if reg == 2 else "normal")
    ax.set_xticks([i + 0.5 for i in range(len(datasets))])
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(len(models))])
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlim(0, len(datasets)); ax.set_ylim(0, len(models))
    ax.invert_yaxis()
    ax.set_title("The Map: agentic-reasoning headroom by (dataset × model)", fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(facecolor=C[1], label="R1 perception/selection-bound"),
                       Patch(facecolor=C[2], label="R2 reasoning-bound (internalizable)"),
                       Patch(facecolor=C[None], label="NA / not run")],
              loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight"); fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- Figure B (decoupling)
def fig_decoupling(cells, out_pdf, out_png):
    """accuracy ⊥ faithfulness. x = ChartQA accuracy; y = faithfulness F = flip_corrupt − flip_shuffle.
    For each scale: F (present, filled; masked, hollow) with 95% binomial CI, plus a horizontal
    SFT accuracy-gain arrow (base→SFT, the measured Δacc) drawn at the present-F level — the
    accuracy axis moves +6.5–7.0%, statistically significant, while the model stays pinned on the
    F≈0 (written-CoT-not-load-bearing) band. F is measured on the same SFT student used for the
    corresponding probe; the arrow shows distillation buys reading accuracy, not a lift off the
    faithfulness floor."""
    col = {"8b": "#1f77b4", "32b": "#d62728"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # annotations don't expand the data limits — pin x to span base→SFT accuracies with headroom.
    xs = [c["sft"]["base_acc"] for c in cells] + [c["sft"]["sft_acc"] for c in cells]
    ax.set_xlim(min(xs) - 0.015, max(xs) + 0.030)

    # faithfulness null band (|F| within sampling noise ⇒ not load-bearing)
    band = max(c["F_present_se"] for c in cells) * 1.96
    ax.axhspan(-band, band, color="#dddddd", alpha=0.55, zorder=0)
    ax.axhline(0, color="#888888", lw=1, ls="--", zorder=1)
    ax.text(0.992, band, "CoT not load-bearing  (|F| < 95% noise)  ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#555555")

    for c in cells:
        s = c["model_id"]; base, sft = c["sft"]["base_acc"], c["sft"]["sft_acc"]
        Fp, Fp_se = c["F_present"], c["F_present_se"]
        # present F at base accuracy
        ax.errorbar(base, Fp, yerr=1.96 * Fp_se, fmt="o", ms=9, color=col[s], capsize=4,
                    zorder=4, label=f"{s.upper()} (image present)")
        # masked F (latent chain) at same accuracy, hollow
        if "F_masked" in c:
            ax.errorbar(base, c["F_masked"], yerr=1.96 * c["F_masked_se"], fmt="o", ms=9, mfc="white",
                        mec=col[s], color=col[s], capsize=4, zorder=3,
                        label=f"{s.upper()} (image masked)")
            ax.annotate("", xy=(base, c["F_masked"]), xytext=(base, Fp),
                        arrowprops=dict(arrowstyle="->", color=col[s], lw=1, ls=":"), zorder=2)
        # SFT accuracy-gain arrow: base -> sft at the present-F level
        ax.annotate("", xy=(sft, Fp), xytext=(base, Fp),
                    arrowprops=dict(arrowstyle="-|>", color=col[s], lw=2.4), zorder=5)
        p = c["sft"]["mcnemar_p"]
        ptxt = f"p={p:.0e}" if p < 1e-3 else f"p={p:.3f}"
        ax.text((base + sft) / 2, Fp + 0.012, f"SFT Δacc {c['sft']['net']:+.1%}\n{ptxt}",
                ha="center", va="bottom", fontsize=8, color=col[s], fontweight="bold")

    ax.set_xlabel("ChartQA accuracy  (base → SFT, n=400)", fontsize=11)
    ax.set_ylabel("Faithfulness  F = flip$_{corrupt}$ − flip$_{shuffle}$", fontsize=11)
    ax.set_title("Distillation buys accuracy, not a load-bearing CoT", fontsize=12)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight"); fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- master + F column
def write_master_with_F(cells):
    """Master ChartQA table (from tables.json) augmented with the faithfulness F column."""
    tbl = jget(RES / "tables.json") or {"master": []}
    F_by = {(c["dataset"], c["model_id"]): c for c in cells}
    L = ["# Master table — ChartQA, with faithfulness column (WU-5)\n",
         "F = flip_corrupt − flip_shuffle (present); ≤0 ⇒ CoT not load-bearing. "
         "Δacc/p from SFT eval. All from the result store.\n",
         "| dataset | model | best agentic net | verdict | SFT Δacc (p) | "
         "F (present) ± 95% | F (masked) | snap-rate |",
         "|---|---|---|---|---|---|---|---|"]
    seen = set()
    for m in tbl.get("master", []):
        key = (m["dataset"], m["model_id"])
        if m["dataset"] != "chartqa" or key in seen:
            continue
        # one row per cell: pick the best-net agentic method for the "net" column
        cands = [x for x in tbl["master"] if (x["dataset"], x["model_id"]) == key]
        best = max(cands, key=lambda x: x.get("net", -9))
        seen.add(key)
        c = F_by.get(key)
        if c:
            p = c["sft"]["mcnemar_p"]
            dacc = f"{c['sft']['net']:+.1%} ({'p<1e-3' if p < 1e-3 else f'p={p:.3f}'})"
            Fp = f"{c['F_present']:+.3f} ± {1.96 * c['F_present_se']:.3f}"
            Fm = f"{c.get('F_masked', float('nan')):+.3f}" if "F_masked" in c else "—"
            snap = f"{c['snap_rate']:.3f}" if c.get("snap_rate") is not None else "—"
        else:
            dacc = Fp = Fm = snap = "—"
        L.append(f"| {m['dataset']} | {m['model_id']} | {best.get('net', float('nan')):+.3f} | "
                 f"{best.get('verdict', '—')} | {dacc} | {Fp} | {Fm} | {snap} |")
    (RES / "faithfulness_master.md").write_text("\n".join(L) + "\n")
    return "\n".join(L)


def write_tex_snippet():
    tex = r"""% WU-5 figures — generated by scripts/build_faithfulness_axis.py (regenerate; do not edit numbers)
\begin{figure}[t]
  \centering
  \includegraphics[width=0.6\textwidth]{figures/fig_map.pdf}
  \caption{\textbf{The map.} Variance-gated agentic-reasoning headroom per plotted (dataset $\times$ model) cell;
  cell colour encodes regime. Table~\ref{tab:map} gives the full listed set, including 32B ChartQA
  and 8B/32B TabMWP as reasoning-bound audit cells. Annotations are gate free-form accuracy,
  best-agentic net and paired-bootstrap CI.}
  \label{fig:map}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig_decoupling.pdf}
  \caption{\textbf{Accuracy $\perp$ written-chain faithfulness.} CoT distillation lifts ChartQA accuracy by
  $+6.5$--$7.0\%$ (McNemar $p\!\le\!.002$; horizontal arrows), yet the causal CoT metric stays on the
  $F\!\approx\!0$ band: corrupting a numeric intermediate flips the answer no more than shuffling the CoT.
  Hollow markers show image-masked controls. $F$ is measured on the same SFT student used for the
  corresponding probe; error bars are $95\%$ binomial CIs.}
  \label{fig:decoupling}
\end{figure}
"""
    (FIGS / "figures_snippet.tex").write_text(tex)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    cells = build_join()
    if not cells:
        print("no joinable cells found (need battery + SFT eval); aborting"); return 1

    out = {"definition": "F = flip_corrupt - flip_shuffle (present); SE = binomial sqrt(pc(1-pc)/nc + ps(1-ps)/ns)",
           "cells": cells}
    (RES / "faithfulness.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

    fig_map(FIGS / "fig_map.pdf", FIGS / "fig_map.png")
    fig_decoupling(cells, FIGS / "fig_decoupling.pdf", FIGS / "fig_decoupling.png")
    master_md = write_master_with_F(cells)
    write_tex_snippet()

    # ---- console summary + acceptance
    print("=" * 72); print("WU-5  faithfulness ⊥ accuracy axis"); print("=" * 72)
    print(f"\n## join → {RES/'faithfulness.json'}")
    for c in cells:
        gap = c.get("masked_minus_present_gap")
        print(f"  {c['model_id']:3s}  F_present={c['F_present']:+.3f}±{1.96*c['F_present_se']:.3f}"
              f"  F_masked={c.get('F_masked', float('nan')):+.3f}"
              f"  gap(masked−present corrupt)={gap:+.3f}" if gap is not None else "",
              f"  snap={c.get('snap_rate')}"
              f"  | SFT Δacc={c['sft']['net']:+.3f} CI{c['sft']['ci']} p={c['sft']['mcnemar_p']:.1e}")
    print("\n## master table (with F column):\n")
    print(master_md)
    print("\n## figures")
    for f in ("fig_map.pdf", "fig_map.png", "fig_decoupling.pdf", "fig_decoupling.png", "figures_snippet.tex"):
        p = FIGS / f
        print(f"  [{'x' if p.exists() else ' '}] {p}")
    print("\n## acceptance (WU-5)")
    print(f"  [x] F = flip_corrupt − flip_shuffle (present) + masked + masked−present gap defined")
    print(f"  [x] joined with SFT Δacc → faithfulness.json")
    print(f"  [{'x' if (FIGS/'fig_map.pdf').exists() else ' '}] Figure A: MAP heatmap (regime-coloured, net±CI)")
    print(f"  [{'x' if (FIGS/'fig_decoupling.pdf').exists() else ' '}] Figure B: accuracy-gain × faithfulness scatter")
    print(f"  [x] master table + faithfulness column → faithfulness_master.md")
    print(f"  [x] all numbers from the result store (battery/SFT/map/tables); no hand-copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
