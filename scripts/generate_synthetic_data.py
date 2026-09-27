"""Generate privacy-safe telecom-shaped data for pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    customer_id = np.arange(1, rows + 1)
    months = rng.integers(1, 72, rows)
    eqpdays = rng.integers(10, 1_800, rows)
    avgrev = np.maximum(10, rng.normal(62, 24, rows))
    mou_mean = np.maximum(0, rng.gamma(2.5, 180, rows))
    change_mou = rng.normal(0, 120, rows)
    drop_vce = np.maximum(0, rng.gamma(1.3, 2.2, rows))

    logit = (
        -0.60
        - 0.015 * months
        + 0.0012 * eqpdays
        - 0.007 * change_mou
        - 0.0008 * mou_mean
        + 0.08 * drop_vce
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    client = pd.DataFrame(
        {
            "Customer_ID": customer_id,
            "avgrev": avgrev.round(2),
            "eqpdays": eqpdays,
            "new_cell": rng.choice(["Y", "N"], rows),
            "dualband": rng.choice(["Y", "N"], rows),
            "ethnic": rng.choice(["A", "B", "C"], rows),
            "income": rng.integers(1, 10, rows),
            "marital": rng.choice(["M", "S", "U"], rows),
            "area": rng.choice(["NORTH", "SOUTH", "CENTRAL"], rows),
            "creditcd": rng.choice(["Y", "N"], rows),
        }
    )
    record = pd.DataFrame(
        {
            "Customer_ID": customer_id,
            "churn": churn,
            "months": months,
            "mou_Mean": mou_mean.round(2),
            "change_mou": change_mou.round(2),
            "drop_vce_Mean": drop_vce.round(2),
            "rev_Mean": (avgrev + rng.normal(0, 5, rows)).round(2),
            "plan_type": rng.choice(["basic", "plus", "premium"], rows),
        }
    )

    # Exercise missing-value handling without exposing real records.
    for frame, column in ((client, "avgrev"), (record, "change_mou")):
        missing_rows = rng.choice(rows, size=max(1, rows // 50), replace=False)
        frame.loc[missing_rows, column] = np.nan
    return client, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--rows", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 200:
        parser.error("--rows must be at least 200")

    client, record = generate(args.rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client.to_csv(args.output_dir / "Client.csv", index=False)
    record.to_csv(args.output_dir / "Record.csv", index=False)
    print(f"Wrote {len(client):,} synthetic rows to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
