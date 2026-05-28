"""Tier classification for distillation trajectories (Phase B, spec §5.1).

A teacher trajectory is graded into one of three tiers:

* tier_1 — clean, judge-correct, no runtime guard fired. The only tier used by
  default for SFT.
* tier_2 — judge-correct but a runtime guard fired (dedup/stall/salvage/cap/
  empty). Off by default; opt in with --include-tier-2 after manual spot-check.
* tier_3 — discard (teacher itself wrong, or terminated via cap/empty).

Note: forced_answer / forced_verify are NOT runtime guards — they are the normal
STOP-DISCIPLINE flow. The forced *messages* are still excluded as training
targets in app.distill_format, but their presence does not demote a trajectory.
"""

from __future__ import annotations

from typing import Any

RUNTIME_GUARDS = {"dedup", "stall", "salvage", "cap", "empty"}

TIER_1 = "tier_1"
TIER_2 = "tier_2"
TIER_3 = "tier_3"


def classify_tier(traj: dict[str, Any]) -> str:
    if not traj.get("judge_correct"):
        return TIER_3
    if traj.get("agent_terminated") in ("cap", "empty"):
        return TIER_3
    guards = set(traj.get("guards_triggered") or [])
    if guards & RUNTIME_GUARDS:
        return TIER_2
    return TIER_1
