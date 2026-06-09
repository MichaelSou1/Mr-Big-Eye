from __future__ import annotations

import argparse
import json
import math


Z_BY_POWER = {
    0.80: 0.841621,
    0.85: 1.036433,
    0.90: 1.281552,
    0.95: 1.644854,
}
Z_BY_ALPHA_TWO_SIDED = {
    0.10: 1.644854,
    0.05: 1.959964,
    0.01: 2.575829,
}


def required_n_per_arm(
    *,
    treatment_rate: float,
    control_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    if not 0 <= treatment_rate <= 1 or not 0 <= control_rate <= 1:
        raise ValueError("rates must be in [0, 1]")
    effect = abs(treatment_rate - control_rate)
    if effect <= 0:
        raise ValueError("treatment_rate and control_rate must differ")
    z_alpha = Z_BY_ALPHA_TWO_SIDED.get(round(alpha, 2), 1.959964)
    z_power = Z_BY_POWER.get(round(power, 2), 0.841621)
    pooled = (treatment_rate + control_rate) / 2.0
    term1 = z_alpha * math.sqrt(2 * pooled * (1 - pooled))
    term2 = z_power * math.sqrt(
        treatment_rate * (1 - treatment_rate) + control_rate * (1 - control_rate)
    )
    return math.ceil(((term1 + term2) ** 2) / (effect ** 2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-proportion power analysis for CoT flip rates.")
    parser.add_argument("--treatment-rate", type=float, required=True)
    parser.add_argument("--control-rate", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    args = parser.parse_args()

    n = required_n_per_arm(
        treatment_rate=args.treatment_rate,
        control_rate=args.control_rate,
        alpha=args.alpha,
        power=args.power,
    )
    print(
        json.dumps(
            {
                "n_per_arm": n,
                "total_two_arm": n * 2,
                "treatment_rate": args.treatment_rate,
                "control_rate": args.control_rate,
                "alpha": args.alpha,
                "power": args.power,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
