#!/usr/bin/env python
"""N1 (B2) — build the load-bearing-chain CURRICULUM (and the matched vanilla control)
from FinQA, with the gold-program oracle that N3 showed is required.

Why FinQA: it ships a gold *executable reasoning program* per example, which is the only
clean way to (a) keep problems whose answer genuinely requires a multi-step chain and
(b) verify, by re-execution, that corrupting a specific operand actually *flips* the answer
(redundancy-aware probe ②). Free-form numeric answers (same relaxed grader, no MCQ), table
available as TEXT (for the ⑤ text-teacher), within the paper's table-QA scope.

Pipeline per kept example:
  1. parse + execute the gold program; require executor == gold (sanity).
  2. ③ filters: >=MIN_OPS ops; >=2 distinct literal operands; >=1 corruption-flippable
     literal (re-execute with operand*2+7 -> final changes); all literals present in the
     table (rendered image is self-sufficient); gold != any single table cell
     (not single-cell-readable).
  3. ② tag: load_bearing_values = the flippable literal operands (the probe corrupts one of
     THESE in the student's CoT, never a random number).
  4. render the table to an image finqa_<i>.png (the VLM student's input).
  5. teacher CoT (⑤): 'gold_program' = deterministic NL render of the gold program
     (no API, provably load-bearing); 'deepseek' = text reasoner reads table-as-text +
     question, consistency-filtered (final==gold) AND must contain a load_bearing_value.

Splits: dev -> curriculum-train, test -> eval-probe source (disjoint, both program-backed).
--filter strict = the curriculum (③); --filter none = matched-size random vanilla control.

Raw data (with program) lives in data/distill/finqa/raw/{dev,test}_full.json (fetched from
the jsDelivr GitHub CDN; HF auto-parquet drops `program`). Run in env `mbe-up`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ----------------------- FinQA program executor (the ② oracle) -----------------------
OPS = {"add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
       "multiply": lambda a, b: a * b, "divide": lambda a, b: (a / b if b != 0 else None),
       "exp": lambda a, b: a ** b}


def _num(tok):
    t = str(tok).strip().replace("$", "").replace(",", "").replace("%", "")
    if t.startswith("const_"):
        t = t[6:].replace("m1", "-1").replace("_", ".")
    try:
        return float(t)
    except ValueError:
        return None


def parse_steps(prog):
    return [(m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()])
            for m in re.finditer(r"([a-z_]+)\(([^)]*)\)", str(prog))]


def execute(steps):
    vals = []
    for op, args in steps:
        if op not in OPS:
            return None
        rs = []
        for a in args:
            if a.startswith("#"):
                k = int(a[1:])
                if k >= len(vals) or vals[k] is None:
                    return None
                rs.append(vals[k])
            else:
                v = _num(a)
                if v is None:
                    return None
                rs.append(v)
        if len(rs) != 2:
            return None
        r = OPS[op](*rs)
        if r is None:
            return None
        vals.append(r)
    return vals[-1] if vals else None


def close(a, b, tol=0.02):
    try:
        a = float(a); b = float(b)
        return abs(a - b) <= abs(b) * tol + 1e-6
    except (TypeError, ValueError):
        return False


def literal_ops(steps):
    """Non-#ref, non-const numeric operands (the values pulled from the table)."""
    out = []
    for op, args in steps:
        for a in args:
            if a.startswith("#") or a.startswith("const_"):
                continue
            v = _num(a)
            if v is not None:
                out.append(v)
    return out


def flippable_operands(steps, res):
    """Return distinct literal values whose corruption (v*2+7) changes the final answer."""
    out = []
    for si, (op, args) in enumerate(steps):
        for ai, a in enumerate(args):
            if a.startswith("#") or _num(a) is None:
                continue
            s2 = [(o, list(ar)) for o, ar in steps]
            s2[si][1][ai] = str(_num(a) * 2 + 7)
            r2 = execute(s2)
            if r2 is not None and not close(r2, res):
                out.append(_num(a))
    # distinct, stable order
    seen, uniq = set(), []
    for v in out:
        key = round(v, 6)
        if key not in seen:
            seen.add(key); uniq.append(v)
    return uniq


def table_nums(tbl):
    out = []
    for row in tbl:
        for c in row:
            m = re.search(r"-?\d[\d,]*\.?\d*", str(c).replace("(", "-").replace("$", ""))
            if m:
                v = _num(m.group(0))
                if v is not None:
                    out.append(v)
    return out


def table_to_text(tbl):
    return "\n".join(" | ".join(str(c) for c in row) for row in tbl)


def passes_curriculum(rec, min_ops):
    """Apply ③ filters. Returns (ok, info) where info has program/res/load_bearing/n_ops."""
    qa = rec.get("qa", {})
    prog = qa.get("program", ""); gold = qa.get("exe_ans")
    steps = parse_steps(prog)
    if not steps or any(op not in OPS for op, _ in steps):
        return False, None
    try:
        float(gold)
    except (TypeError, ValueError):
        return False, None
    res = execute(steps)
    if res is None or not close(res, gold):
        return False, None
    if len(steps) < min_ops:
        return False, None
    lits = literal_ops(steps)
    if len(set(round(x, 6) for x in lits)) < 2:
        return False, None
    lb = flippable_operands(steps, res)
    if not lb:
        return False, None
    tn = table_nums(rec.get("table", []))
    if not all(any(close(l, t) for t in tn) for l in lits):
        return False, None                       # (e) image self-sufficient
    if any(close(gold, t) for t in tn):
        return False, None                       # (d) not single-cell-readable
    return True, {"program": prog, "res": res, "load_bearing_values": lb, "n_ops": len(steps)}


# ----------------------- teacher CoT (⑤) -----------------------
def gold_program_cot(steps):
    """Deterministic NL render of the gold program (provably load-bearing, no API)."""
    vals, lines = [], []
    TMPL = {"add": "add {0} and {1}", "subtract": "subtract {1} from {0}",
            "multiply": "multiply {0} by {1}", "divide": "divide {0} by {1}",
            "exp": "raise {0} to the power {1}"}
    for i, (op, args) in enumerate(steps):
        rs = [f"step {int(a[1:])+1}'s result" if a.startswith("#") else a.replace("const_", "")
              for a in args]
        r = execute(steps[:i + 1])
        lines.append(f"Step {i+1}: {TMPL[op].format(*rs)} to get {round(r,4)}.")
        vals.append(r)
    return " ".join(lines), round(vals[-1], 4)


_TEACHER_SYS = (
    "You are given a data table (as text) and a question. Read the needed values from the "
    "table and reason step by step, showing each arithmetic operation and its result. "
    "End with a line exactly 'ANSWER: <number>'. Be concise."
)


_TEACHER_HINT_SYS = (
    "You are given a data table (as text), a question, and the CORRECT final answer. Write a "
    "concise, natural step-by-step calculation that reads the needed values from the table and "
    "arrives at exactly that answer, showing each arithmetic operation and its intermediate result. "
    "Do NOT mention that the answer was given; present it as your own derivation. "
    "End with a line exactly 'ANSWER: <the given answer>'."
)


def _parse_cot_ans(out):
    m = re.search(r"ANSWER:\s*(.+)", out, re.IGNORECASE)
    ans = m.group(1).strip().splitlines()[0].strip() if m else ""
    cot = out[:m.start()].strip() if m else out.strip()
    return cot, ans


def deepseek_cot(table_text, question):
    """Unconditioned text-reasoner teacher (reads table-as-text, must compute correctly)."""
    from app.distill.methods import orch
    msg = [{"role": "system", "content": _TEACHER_SYS},
           {"role": "user", "content": f"Table:\n{table_text}\n\nQuestion: {question}"}]
    return _parse_cot_ans(orch(msg, temp=0.2, max_tokens=512))


def deepseek_hint_cot(table_text, question, gold, lb_values):
    """Answer-conditioned rationalization: the teacher is told the gold answer (and the
    load-bearing operand values, to anchor the chain on real table cells) and writes a FLUENT
    load-bearing chain reaching it. Removes the teacher-quality confound (the gold-program arm's
    purpose) while staying naturalistic — isolates 'does a fluent, correct, load-bearing teacher
    chain transfer load-bearing-ness?'. No probe leakage: the student gets no privileged info at test."""
    from app.distill.methods import orch
    hint = (f"Table:\n{table_text}\n\nQuestion: {question}\n\nCorrect final answer: {gold}"
            + (f"\nKey table values to use: {', '.join(str(v) for v in lb_values)}" if lb_values else ""))
    return _parse_cot_ans(orch([{"role": "system", "content": _TEACHER_HINT_SYS},
                                {"role": "user", "content": hint}], temp=0.2, max_tokens=512))


# ----------------------- table -> image -----------------------
def render_table(tbl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [[str(c) for c in r] for r in tbl if len(r)]
    if not rows:
        return False
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    fig_w = min(2 + 1.6 * ncol, 16); fig_h = min(0.5 + 0.4 * len(rows), 14)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.axis("off")
    t = ax.table(cellText=rows, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.3)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return True


# ----------------------- main -----------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev",
                    help="dev -> curriculum-train; test -> eval-probe source")
    ap.add_argument("--raw", default="data/distill/finqa/raw")
    ap.add_argument("--filter", choices=["strict", "none"], default="strict",
                    help="strict = ③ curriculum; none = matched random vanilla control")
    ap.add_argument("--teacher", choices=["gold_program", "deepseek", "deepseek_hint"],
                    default="gold_program")
    ap.add_argument("--min-ops", type=int, default=2)
    ap.add_argument("--target", type=int, default=180, help="max kept examples")
    ap.add_argument("--img-dir", default=None, help="default: finqa_<split>_images")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", type=int, default=0, help="parse+filter only on N, no render/teacher")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    raw = json.load(open(f"{args.raw}/{args.split}_full.json"))
    img_dir = Path(args.img_dir or f"/home/gpus/mbe_data/finqa_{args.split}_images")
    out = Path(args.out or f"data/distill/finqa/curriculum_{args.split}_{args.filter}.jsonl")

    # candidate pool
    if args.filter == "strict":
        pool = []
        for rec in raw:
            ok, info = passes_curriculum(rec, args.min_ops)
            if ok:
                pool.append((rec, info))
    else:  # vanilla: executable numeric, NO ③ — random matched sample
        pool = []
        for rec in raw:
            qa = rec.get("qa", {}); steps = parse_steps(qa.get("program", ""))
            if not steps or any(op not in OPS for op, _ in steps):
                continue
            res = execute(steps)
            if res is None or not close(res, qa.get("exe_ans")):
                continue
            pool.append((rec, {"program": qa.get("program"), "res": res,
                               "load_bearing_values": [], "n_ops": len(steps)}))
        rng.shuffle(pool)
    print(f"[{args.split}/{args.filter}] candidate pool: {len(pool)} (target {args.target})", flush=True)

    if args.dry_run:
        for rec, info in pool[:args.dry_run]:
            print(f"  {rec['id']}: ops={info['n_ops']} prog={info['program']} "
                  f"lb={info['load_bearing_values']} gold={rec['qa'].get('exe_ans')}")
        print(f"DRY-RUN: {len(pool)} would pass; showed {min(args.dry_run,len(pool))}")
        return 0

    img_dir.mkdir(parents=True, exist_ok=True); out.parent.mkdir(parents=True, exist_ok=True)
    kept = attempted = drift = 0
    with open(out, "w", encoding="utf-8") as fh:
        for i, (rec, info) in enumerate(pool):
            if kept >= args.target:
                break
            attempted += 1
            qa = rec.get("qa", {}); q = str(qa.get("question", "")); gold = str(qa.get("exe_ans"))
            steps = parse_steps(info["program"])
            img_path = img_dir / f"finqa_{kept}.png"
            try:
                if not render_table(rec.get("table", []), img_path):
                    continue
            except Exception as e:
                print(f"  render err {str(e)[:50]}", flush=True); continue
            if args.teacher == "gold_program":
                cot, ans = gold_program_cot(steps)
            else:
                try:
                    if args.teacher == "deepseek_hint":
                        cot, ans = deepseek_hint_cot(table_to_text(rec.get("table", [])), q,
                                                     gold, info["load_bearing_values"])
                    else:
                        cot, ans = deepseek_cot(table_to_text(rec.get("table", [])), q)
                except Exception as e:
                    print(f"  teacher err {str(e)[:50]}", flush=True); continue
                if not (cot and ans and close(ans, gold)):
                    drift += 1; continue
                # require the CoT to contain a load-bearing value (so the probe has a target)
                if args.filter == "strict":
                    cot_nums = [_num(x) for x in re.findall(r"-?\d[\d,]*\.?\d*", cot)]
                    if not any(any(close(lb, cn) for cn in cot_nums if cn is not None)
                               for lb in info["load_bearing_values"]):
                        continue
            fh.write(json.dumps({
                "case_id": f"finqa-{kept}", "finqa_id": rec.get("id"),
                "image_path": str(img_path), "question": q,
                "cot": cot, "answer": str(ans), "gold": gold,
                "program": info["program"], "n_ops": info["n_ops"],
                "load_bearing_values": info["load_bearing_values"],
            }, ensure_ascii=False) + "\n")
            kept += 1
    print(f"DONE [{args.split}/{args.filter}/{args.teacher}]: kept {kept} "
          f"(attempted {attempted}, teacher-drift dropped {drift}) -> {out}", flush=True)
    print(f"  images -> {img_dir}/finqa_<i>.png", flush=True)
    return 0 if kept >= 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
