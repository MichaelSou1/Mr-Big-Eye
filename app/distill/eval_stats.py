"""Statistics for the variance gate (Spec §1.3, §2). Stdlib + numpy only.

Primary test is a PAIRED bootstrap over cases (methods evaluated on the same cases,
same frames). McNemar is the secondary sanity check. Power uses the existing
two-proportion machinery in app/distill/power_analysis.py, inverted to "given n →
minimum detectable net".
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from app.distill.power_analysis import Z_BY_ALPHA_TWO_SIDED, Z_BY_POWER


def paired_bootstrap_net(
    free_correct: Sequence[bool],
    method_correct: Sequence[bool],
    *,
    B: int = 10000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap over cases for net = acc(method) - acc(free_form).

    Returns net (point), 95% CI, and p_le0 / p_ge0 = fraction of resamples with
    net ≤ 0 / ≥ 0 (a two-sided-ish read on whether the CI excludes 0).
    """
    f = np.asarray(free_correct, dtype=float)
    m = np.asarray(method_correct, dtype=float)
    n = len(f)
    if n == 0 or len(m) != n:
        return {"n": n, "net": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_le0": 1.0, "p_ge0": 1.0,
                "gain": 0, "lost": 0, "excludes_0": False}
    diff = m - f                      # per-case net contribution
    net = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = diff[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    gain = int(((m == 1) & (f == 0)).sum())
    lost = int(((m == 0) & (f == 1)).sum())
    return {
        "n": n,
        "net": net,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_le0": float((boot <= 0).mean()),
        "p_ge0": float((boot >= 0).mean()),
        "gain": gain,
        "lost": lost,
        "excludes_0": bool(ci_lo > 0 or ci_hi < 0),
    }


def mcnemar(gain: int, lost: int) -> dict:
    """McNemar exact-ish test on discordant pairs (gain vs lost). Returns chi2 + p (approx)."""
    b, c = gain, lost
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)        # with continuity correction
    # survival of chi2 with df=1 via erfc
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": float(chi2), "p": float(p)}


def min_detectable_net(p_baseline: float, n: int, *, power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest |net| detectable at given n (per arm = paired n) with target power.

    Solves the two-proportion sample-size formula for the effect, treating the paired
    set size as n. Uses the same z-tables as power_analysis.required_n_per_arm.
    """
    z_a = Z_BY_ALPHA_TWO_SIDED.get(round(alpha, 2), 1.959964)
    z_b = Z_BY_POWER.get(round(power, 2), 0.841621)
    p = min(max(p_baseline, 1e-3), 1 - 1e-3)
    # n ≈ (z_a*sqrt(2 p (1-p)) + z_b*sqrt(p1(1-p1)+p2(1-p2)))^2 / effect^2 ; approximate the
    # variance terms at p → effect ≈ (z_a+z_b)*sqrt(2 p (1-p) / n).
    return float((z_a + z_b) * math.sqrt(2 * p * (1 - p) / max(n, 1)))


def agg_seed_nets(per_seed_nets: Sequence[float]) -> dict:
    """Seed-mean net + seed std (σ_decode) for stochastic methods (Spec §1.1)."""
    a = np.asarray(per_seed_nets, dtype=float)
    if a.size == 0:
        return {"k": 0, "net_mean": 0.0, "net_std": 0.0}
    return {"k": int(a.size), "net_mean": float(a.mean()), "net_std": float(a.std(ddof=1) if a.size > 1 else 0.0)}
