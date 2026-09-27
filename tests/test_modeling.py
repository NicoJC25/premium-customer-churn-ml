import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_model.modeling import train_and_compare


def test_quick_training_returns_finite_metrics() -> None:
    rng = np.random.default_rng(42)
    rows = 500
    numeric = rng.normal(size=rows)
    category = rng.choice(["a", "b", "c"], size=rows)
    probability = 1 / (1 + np.exp(-(0.9 * numeric + (category == "c") * 0.6)))
    target = pd.Series(rng.binomial(1, probability), name="churn")
    features = pd.DataFrame({"numeric": numeric, "category": category})
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, stratify=target, test_size=0.25, random_state=42
    )

    results = train_and_compare(
        X_train,
        y_train,
        X_test,
        y_test,
        quick=True,
        random_state=42,
    )

    assert {result.name for result in results} == {
        "dummy",
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    for result in results:
        assert 0 <= result.metrics["roc_auc"] <= 1
        assert np.isfinite(list(result.metrics.values())).all()
