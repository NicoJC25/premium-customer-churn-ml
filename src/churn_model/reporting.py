"""Save privacy-safe aggregate metrics and visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn_model.modeling import ModelResult, feature_importance


def save_reports(
    results: list[ModelResult],
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> ModelResult:
    """Save aggregate artifacts and return the best non-dummy model."""

    output = Path(output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    candidates = [result for result in results if result.name != "dummy"]
    best = max(candidates, key=lambda result: result.metrics["roc_auc"])

    comparison = pd.DataFrame(
        [{"model": result.name, **result.metrics} for result in results]
    ).sort_values("roc_auc", ascending=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)

    payload = {
        "dataset": metadata,
        "best_model": best.name,
        "best_params": best.best_params,
        "models": {result.name: result.metrics for result in results},
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    importance = feature_importance(best)
    importance.to_csv(output / "feature_importance.csv", index=False)
    joblib.dump(best.estimator, output / "best_model.joblib")

    _plot_comparison(comparison, figures / "model_comparison.png")
    _plot_roc(results, figures / "roc_curves.png")
    _plot_confusion(best, figures / "confusion_matrix.png")
    if not importance.empty:
        _plot_importance(importance.head(15), figures / "feature_importance.png")
    return best


def _plot_comparison(comparison: pd.DataFrame, path: Path) -> None:
    ordered = comparison.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(ordered["model"], ordered["roc_auc"], color="#2563eb")
    ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=1, label="Random ranking")
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Holdout ROC-AUC")
    ax.set_title("Model comparison")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in ordered["roc_auc"]], padding=4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_roc(results: list[ModelResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        ax.plot(
            result.fpr,
            result.tpr,
            linewidth=2,
            label=f"{result.name} ({result.metrics['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Holdout ROC curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confusion(result: ModelResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    matrix = result.confusion
    image = ax.imshow(matrix, cmap="Blues")
    for row, column in np.ndindex(matrix.shape):
        ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Retained", "Churned"])
    ax.set_yticks([0, 1], labels=["Retained", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{result.name} confusion matrix (threshold = 0.5)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_importance(importance: pd.DataFrame, path: Path) -> None:
    ordered = importance.sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(ordered["feature"], ordered["importance"], color="#0f766e")
    ax.set_xlabel("Model importance")
    ax.set_title("Most influential model features")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
