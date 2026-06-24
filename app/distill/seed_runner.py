"""Seed-loop harness + append-only result store (Spec §7).

A `method_fn(case, seed) -> {"free_answer", "method_answer"}` is run across K seeds;
each output is graded text-aware (eval_common), the paired bootstrap net (free vs
method) is computed per seed and aggregated, and every (setting, seed) row is appended
to the result store keyed by a config fingerprint. Tables regenerate from the store.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from app.distill.eval_common import case_set_hash, config_fingerprint, grade_textaware
from app.distill.eval_stats import agg_seed_nets, paired_bootstrap_net

# Default store path; overridable via MBE_RESULTS_PATH so two gate processes (e.g. a 4B run on
# one GPU and an 8B run on another) can write to separate files concurrently without racing on
# append (the per-case rows exceed PIPE_BUF, so concurrent appends to one file could interleave).
# Merge the side files back into the canonical store afterwards (cat — fingerprints disambiguate).
RESULTS_PATH = Path(os.environ.get("MBE_RESULTS_PATH", "data/distill/results/results.jsonl"))


def append_result(row: dict[str, Any], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_results(path: Path = RESULTS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def run_method_over_seeds(
    *,
    dataset: str,
    model_id: str,
    method: str,
    cases: list[dict],                       # each: {case_id, question, gold, ...}
    method_fn: Callable[[dict, int], dict],  # (case, seed) -> {free_answer, method_answer}
    seeds: list[int],
    decode: dict[str, Any],
    n_frames: int,
    store: bool = True,
    progress: bool = True,
    concurrency: int = 1,
) -> dict:
    """Run `method` across seeds; returns aggregated nets + per-seed bootstrap; stores rows.

    concurrency>1 runs the per-case method_fn over a thread pool (each call does its own
    asyncio.run, so threads are independent) — the DeepSeek API and vllm both serve
    concurrent requests, so this is the main throughput lever for the orch methods.
    """
    from concurrent.futures import ThreadPoolExecutor

    split_hash = case_set_hash([c["case_id"] for c in cases])

    def grade_one(case, seed):
        out = method_fn(case, seed)
        fg = grade_textaware(case["question"], str(case["gold"]), out.get("free_answer", ""))
        mg = grade_textaware(case["question"], str(case["gold"]), out.get("method_answer", ""))
        return {"case_id": case["case_id"], "free_correct": fg["correct"],
                "method_correct": mg["correct"]}

    per_seed = []
    for seed in seeds:
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                rows = list(ex.map(lambda c: grade_one(c, seed), cases))  # preserves order
        else:
            rows = []
            for i, case in enumerate(cases):
                rows.append(grade_one(case, seed))
                if progress and (i + 1) % 20 == 0:
                    print(f"  [{method} seed{seed}] {i+1}/{len(cases)}", flush=True)
        free_ok = [int(r["free_correct"]) for r in rows]
        meth_ok = [int(r["method_correct"]) for r in rows]
        boot = paired_bootstrap_net(free_ok, meth_ok)
        fp = config_fingerprint(dataset=dataset, split_hash=split_hash, model_id=model_id,
                                method=method, n_frames=n_frames, seed=seed, **decode)
        row = {"dataset": dataset, "model_id": model_id, "method": method, "seed": seed,
               "n": len(cases), "free_acc": sum(free_ok) / len(cases),
               "method_acc": sum(meth_ok) / len(cases), "bootstrap": boot,
               "fingerprint": fp, "rows": rows}
        if store:
            append_result(row)
        per_seed.append(boot["net"])
        print(f"[{model_id}/{dataset}/{method} seed{seed}] free={row['free_acc']:.3f} "
              f"method={row['method_acc']:.3f} net={boot['net']:+.3f} "
              f"CI[{boot['ci_lo']:+.3f},{boot['ci_hi']:+.3f}] gain={boot['gain']} lost={boot['lost']}",
              flush=True)
    agg = agg_seed_nets(per_seed)
    # Verdict = "effect" iff every seed's per-seed bootstrap CI excludes 0 with the SAME sign
    # (a strict, honest bar). Otherwise within-variance. Final paper verdict is recomputed in
    # regen_tables from the stored rows (pooled), this is the live summary.
    signs = {("+" if s > 0 else "-") for s in per_seed}
    verdict = "effect" if (per_seed and len(signs) == 1 and abs(agg["net_mean"]) > 2 * agg["net_std"]) \
        else "within-variance"
    return {"dataset": dataset, "model_id": model_id, "method": method,
            "net_mean": agg["net_mean"], "net_std": agg["net_std"], "k": agg["k"],
            "per_seed_nets": per_seed, "verdict": verdict}
