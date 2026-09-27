from pathlib import Path

import pandas as pd
import pytest

from churn_model.data import (
    SENSITIVE_OR_PROXY_COLUMNS,
    load_and_merge,
    select_model_features,
    select_premium_segment,
)


def _write_sources(directory: Path) -> tuple[Path, Path]:
    client = pd.DataFrame(
        {
            "Customer_ID": range(1, 9),
            "avgrev": [10, 20, 30, 40, 50, 60, 70, 80],
            "ethnic": ["A"] * 8,
            "income": [3, 4, 5, 6, 7, 8, 9, 9],
            "eqpdays": [100, 200, 300, 400, 500, 600, 700, 800],
        }
    )
    record = pd.DataFrame(
        {
            "Customer_ID": range(1, 9),
            "churn": [0, 1, 0, 1, 0, 1, 0, 1],
            "months": [1, 2, 3, 4, 5, 6, 7, 8],
            "plan": ["a", "b"] * 4,
        }
    )
    client_path = directory / "Client.csv"
    record_path = directory / "Record.csv"
    client.to_csv(client_path, index=False)
    record.to_csv(record_path, index=False)
    return client_path, record_path


def test_load_segment_and_sensitive_exclusion(tmp_path: Path) -> None:
    client_path, record_path = _write_sources(tmp_path)
    merged = load_and_merge(client_path, record_path)
    segment, cutoff = select_premium_segment(merged, 0.5)
    selected = select_model_features(segment)

    assert len(merged) == 8
    assert cutoff == pytest.approx(45.0)
    assert len(segment) == 4
    assert not (set(selected.X.columns) & SENSITIVE_OR_PROXY_COLUMNS)
    assert "Customer_ID" not in selected.X
    assert "churn" not in selected.X


def test_duplicate_identifiers_are_rejected(tmp_path: Path) -> None:
    client_path, record_path = _write_sources(tmp_path)
    record = pd.read_csv(record_path)
    record.loc[1, "Customer_ID"] = record.loc[0, "Customer_ID"]
    record.to_csv(record_path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_and_merge(client_path, record_path)
