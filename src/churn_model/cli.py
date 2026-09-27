"""Command-line entry point for training and evaluating churn models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from churn_model.data import load_and_merge, select_model_features, select_premium_segment
from churn_model.modeling import train_and_compare
from churn_model.reporting import save_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=Path, required=True, help="Path to Client.csv")
    parser.add_argument("--record", type=Path, required=True, help="Path to Record.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/latest"),
        help="Directory for aggregate reports and the local model artifact",
    )
    parser.add_argument("--premium-quantile", type=float, default=0.75)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use smaller models for smoke tests")
    parser.add_argument(
        "--tune-xgboost",
        action="store_true",
        help="Tune XGBoost on training folds before holdout evaluation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    merged = load_and_merge(args.client, args.record)
    segment, revenue_cutoff = select_premium_segment(merged, args.premium_quantile)
    selection = select_model_features(segment)

    X_train, X_test, y_train, y_test = train_test_split(
        selection.X,
        selection.y,
        test_size=args.test_size,
        stratify=selection.y,
        random_state=args.random_state,
    )
    results = train_and_compare(
        X_train,
        y_train,
        X_test,
        y_test,
        random_state=args.random_state,
        quick=args.quick,
        tune_xgboost=args.tune_xgboost,
    )

    metadata = {
        "total_rows": int(len(merged)),
        "segment_rows": int(len(segment)),
        "segment_share": float(len(segment) / len(merged)),
        "premium_revenue_quantile": float(args.premium_quantile),
        "premium_revenue_cutoff": revenue_cutoff,
        "churn_rate": float(selection.y.mean()),
        "model_feature_count": int(selection.X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "excluded_sensitive_columns": list(selection.excluded_sensitive),
        "excluded_high_missingness_columns": list(selection.excluded_high_missingness),
        "excluded_constant_columns": list(selection.excluded_constant),
        "random_state": int(args.random_state),
    }
    best = save_reports(results, args.output_dir, metadata)

    summary = {
        "best_model": best.name,
        "holdout_roc_auc": round(best.metrics["roc_auc"], 4),
        "holdout_pr_auc": round(best.metrics["pr_auc"], 4),
        "segment_rows": len(segment),
        "output_dir": str(args.output_dir.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
