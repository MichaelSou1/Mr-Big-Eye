"""Log the 3-way distillation eval (base / pilot / v2) to Weights & Biases.

Reads eval_harness.py report JSON(s) per model, joins each case to its modality via
the eval-cases file, computes overall + per-modality pass_rate (mean±std across
repeat runs), and logs to a dedicated W&B run:
  * a `eval_3way` wandb.Table (model × modality → pass_rate, n, std, runs)
  * summary scalars  <model>/overall, <model>/<modality>  (for W&B panels/compare)
  * grouped bar charts (overall + per modality)
  * an Artifact bundling the raw report JSONs + the cases file (provenance)
It also writes a markdown table for the outcomes doc / resume.

Auth: uses ~/.netrc (no token re-config). Run in an env with wandb (e.g. vlm_dapo):

    conda run -n vlm_dapo python scripts/wandb_eval_report.py \
      --project mbe-distill --run-name distill-v2-3way-eval \
      --cases data/distillation/v2/holdout_all_cases.jsonl \
      --model base  reports/base_run1.json  reports/base_run2.json  reports/base_run3.json \
      --model pilot reports/pilot_run1.json reports/pilot_run2.json reports/pilot_run3.json \
      --model v2    reports/v2_run1.json    reports/v2_run2.json    reports/v2_run3.json \
      --md-out reports/eval_3way.md
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _passed_by_case(report_path: Path) -> dict[str, bool]:
    rep = json.loads(Path(report_path).read_text())
    out: dict[str, bool] = {}
    for r in rep.get("results", []):
        cid = r.get("case_id")
        if cid is not None:
            out[cid] = bool(r.get("passed"))
    return out


def _rate(passed: list[bool]) -> float:
    return sum(passed) / len(passed) if passed else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="mbe-distill")
    ap.add_argument("--run-name", default="distill-v2-3way-eval")
    ap.add_argument("--cases", required=True, help="Eval cases JSONL (question_id + modality_tag).")
    ap.add_argument("--model", action="append", nargs="+", required=True,
                    metavar=("NAME", "REPORT"),
                    help="Model name followed by one or more report JSONs (repeat runs).")
    ap.add_argument("--md-out", default=None, help="Also write a markdown table here.")
    ap.add_argument("--no-wandb", action="store_true", help="Compute + print only; skip W&B.")
    args = ap.parse_args()

    modality = {c["question_id"]: c.get("modality_tag", "?")
                for c in _load_jsonl(Path(args.cases))}
    mods = sorted(set(modality.values()))

    # per model: list of (overall, {modality: rate}) across repeat runs
    per_model_runs: dict[str, list[dict]] = {}
    per_model_n: dict[str, dict[str, int]] = {}
    for group in args.model:
        name, reports = group[0], group[1:]
        if not reports:
            raise SystemExit(f"--model {name} needs at least one report JSON")
        runs = []
        for rp in reports:
            pc = _passed_by_case(Path(rp))
            by_mod: dict[str, list[bool]] = defaultdict(list)
            allp = []
            for cid, ok in pc.items():
                m = modality.get(cid, "?")
                by_mod[m].append(ok); allp.append(ok)
            runs.append({"overall": _rate(allp),
                         **{m: _rate(by_mod.get(m, [])) for m in mods}})
            per_model_n[name] = {m: len(by_mod.get(m, [])) for m in mods}
            per_model_n[name]["overall"] = len(allp)
        per_model_runs[name] = runs

    def agg(name: str, key: str) -> tuple[float, float]:
        vals = [r[key] for r in per_model_runs[name] if r[key] == r[key]]  # drop nan
        if not vals:
            return float("nan"), 0.0
        return (sum(vals) / len(vals), (stats.pstdev(vals) if len(vals) > 1 else 0.0))

    # ---- console + markdown table ----
    cols = ["overall"] + mods
    lines = ["| model | " + " | ".join(f"{c} (n)" for c in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for name in per_model_runs:
        cells = []
        for c in cols:
            mean, sd = agg(name, c)
            n = per_model_n[name].get(c, 0)
            cells.append(f"{mean:.3f}±{sd:.3f} ({n})" if sd else f"{mean:.3f} ({n})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table_md = "\n".join(lines)
    print(table_md)
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text(table_md + "\n")
        print(f"\nwrote {args.md_out}")

    if args.no_wandb:
        return 0

    import wandb
    run = wandb.init(project=args.project, name=args.run_name, job_type="eval")
    tbl = wandb.Table(columns=["model", "modality", "n_cases", "pass_rate", "std", "runs"])
    for name in per_model_runs:
        nruns = len(per_model_runs[name])
        for c in cols:
            mean, sd = agg(name, c)
            tbl.add_data(name, c, per_model_n[name].get(c, 0), mean, sd, nruns)
            run.summary[f"{name}/{c}"] = mean
            if sd:
                run.summary[f"{name}/{c}_std"] = sd
    run.log({"eval_3way": tbl})
    # grouped bar chart per modality (model on x via a Table plot)
    for c in cols:
        bt = wandb.Table(columns=["model", "pass_rate"],
                         data=[[n, agg(n, c)[0]] for n in per_model_runs])
        run.log({f"bar/{c}": wandb.plot.bar(bt, "model", "pass_rate",
                                            title=f"pass_rate — {c}")})
    art = wandb.Artifact("eval_3way_reports", type="eval-report")
    for group in args.model:
        for rp in group[1:]:
            art.add_file(rp)
    art.add_file(args.cases)
    if args.md_out:
        art.add_file(args.md_out)
    run.log_artifact(art)
    run.finish()
    print(f"\nlogged to W&B project={args.project} run={args.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
