"""Data loading, validation, segmentation, and feature selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ID_COLUMN = "Customer_ID"
TARGET_COLUMN = "churn"
REVENUE_COLUMN = "avgrev"

# These fields are excluded to reduce privacy and discriminatory-targeting risk.
# Exclusion does not by itself establish model fairness; see docs/responsible-ai.md.
SENSITIVE_OR_PROXY_COLUMNS = {
    "ethnic",
    "marital",
    "income",
    "adults",
    "numbcars",
    "ownrent",
    "lor",
    "dwlltype",
    "dwllsize",
    "HHstatin",
    "kid0_2",
    "kid3_5",
    "kid6_10",
    "kid11_15",
    "kid16_17",
    "forgntvl",
    "area",
    "prizm_social_one",
    "creditcd",
    "crclscod",
    "asl_flag",
    "infobase",
}


@dataclass(frozen=True)
class FeatureSelection:
    """Feature matrix plus an audit trail of removed columns."""

    X: pd.DataFrame
    y: pd.Series
    excluded_sensitive: tuple[str, ...]
    excluded_high_missingness: tuple[str, ...]
    excluded_constant: tuple[str, ...]


def load_and_merge(client_path: str | Path, record_path: str | Path) -> pd.DataFrame:
    """Load the two source tables and perform a validated one-to-one merge."""

    client = pd.read_csv(client_path)
    record = pd.read_csv(record_path)

    for name, frame in (("Client", client), ("Record", record)):
        if ID_COLUMN not in frame.columns:
            raise ValueError(f"{name}.csv is missing required column {ID_COLUMN!r}.")
        if frame[ID_COLUMN].isna().any():
            raise ValueError(f"{name}.csv contains null customer identifiers.")
        if not frame[ID_COLUMN].is_unique:
            raise ValueError(f"{name}.csv contains duplicate customer identifiers.")

    merged = client.merge(record, on=ID_COLUMN, how="inner", validate="one_to_one")
    required = {ID_COLUMN, TARGET_COLUMN, REVENUE_COLUMN}
    missing = sorted(required.difference(merged.columns))
    if missing:
        raise ValueError(f"Merged data is missing required columns: {missing}")
    if merged.empty:
        raise ValueError("The merge produced no records.")
    if not set(merged[TARGET_COLUMN].dropna().unique()).issubset({0, 1}):
        raise ValueError("The churn target must be binary (0/1).")

    merged[TARGET_COLUMN] = merged[TARGET_COLUMN].astype("int8")
    return merged


def select_premium_segment(
    frame: pd.DataFrame,
    revenue_quantile: float = 0.75,
) -> tuple[pd.DataFrame, float]:
    """Select customers at or above a revenue quantile.

    The original assignment mixed revenue and household composition. This
    version uses revenue alone so that "premium" has a precise business meaning
    and does not depend on family-status attributes.
    """

    if not 0 < revenue_quantile < 1:
        raise ValueError("revenue_quantile must be between 0 and 1.")
    revenue = pd.to_numeric(frame[REVENUE_COLUMN], errors="coerce")
    cutoff = float(revenue.quantile(revenue_quantile))
    segment = frame.loc[revenue >= cutoff].copy()
    if segment[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The selected segment must contain both churn classes.")
    return segment, cutoff


def select_model_features(
    frame: pd.DataFrame,
    max_missing_fraction: float = 0.95,
) -> FeatureSelection:
    """Remove identifiers, sensitive fields, unusable fields, and the target."""

    if not 0 <= max_missing_fraction < 1:
        raise ValueError("max_missing_fraction must be in [0, 1).")

    candidates = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")
    sensitive = tuple(sorted(set(candidates.columns) & SENSITIVE_OR_PROXY_COLUMNS))
    candidates = candidates.drop(columns=list(sensitive), errors="ignore")

    missing_fraction = candidates.isna().mean()
    high_missing = tuple(sorted(missing_fraction[missing_fraction > max_missing_fraction].index))
    candidates = candidates.drop(columns=list(high_missing), errors="ignore")

    constant = tuple(
        sorted(column for column in candidates if candidates[column].nunique(dropna=True) <= 1)
    )
    candidates = candidates.drop(columns=list(constant), errors="ignore")
    if candidates.empty:
        raise ValueError("No usable model features remain after validation.")

    return FeatureSelection(
        X=candidates,
        y=frame[TARGET_COLUMN].copy(),
        excluded_sensitive=sensitive,
        excluded_high_missingness=high_missing,
        excluded_constant=constant,
    )
