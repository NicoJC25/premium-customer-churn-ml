"""Leakage-safe model training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


@dataclass
class ModelResult:
    """Fitted model and holdout evaluation outputs."""

    name: str
    estimator: BaseEstimator
    metrics: dict[str, float]
    y_probability: np.ndarray
    y_prediction: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    confusion: np.ndarray
    best_params: dict[str, Any] | None = None


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing that is fitted only as part of a model pipeline."""

    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
    )


def _pipeline(X: pd.DataFrame, model: BaseEstimator) -> Pipeline:
    return Pipeline(steps=[("preprocess", build_preprocessor(X)), ("model", model)])


def _evaluate(
    name: str,
    estimator: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: dict[str, Any] | None = None,
) -> ModelResult:
    probability = np.asarray(estimator.predict_proba(X_test)[:, 1])
    prediction = (probability >= 0.5).astype("int8")
    fpr, tpr, _ = roc_curve(y_test, probability)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "pr_auc": float(average_precision_score(y_test, probability)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, probability)),
    }
    return ModelResult(
        name=name,
        estimator=estimator,
        metrics=metrics,
        y_probability=probability,
        y_prediction=prediction,
        fpr=fpr,
        tpr=tpr,
        confusion=confusion_matrix(y_test, prediction),
        best_params=best_params,
    )


def train_and_compare(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_state: int = 42,
    quick: bool = False,
    tune_xgboost: bool = False,
) -> list[ModelResult]:
    """Fit baselines and candidate models, then evaluate once on the holdout set."""

    models: list[tuple[str, BaseEstimator]] = [
        ("dummy", DummyClassifier(strategy="prior")),
        (
            "logistic_regression",
            LogisticRegression(max_iter=1_500, random_state=random_state),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=100 if quick else 300,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
    ]

    results: list[ModelResult] = []
    for name, model in models:
        fitted = _pipeline(X_train, model).fit(X_train, y_train)
        results.append(_evaluate(name, fitted, X_test, y_test))

    xgboost = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=100 if quick else 300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=random_state,
    )
    xgb_pipeline = _pipeline(X_train, xgboost)
    best_params: dict[str, Any] | None = None

    if tune_xgboost and not quick:
        search = RandomizedSearchCV(
            estimator=xgb_pipeline,
            param_distributions={
                "model__n_estimators": [200, 300, 450],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__subsample": [0.7, 0.8, 0.9],
                "model__colsample_bytree": [0.7, 0.8, 0.9],
                "model__reg_alpha": [0.0, 0.1, 1.0],
                "model__reg_lambda": [1.0, 2.0, 5.0],
            },
            n_iter=10,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            n_jobs=-1,
            random_state=random_state,
            refit=True,
            verbose=1,
        )
        search.fit(X_train, y_train)
        fitted_xgb = search.best_estimator_
        best_params = {
            key.replace("model__", ""): value
            for key, value in search.best_params_.items()
        }
    else:
        fitted_xgb = xgb_pipeline.fit(X_train, y_train)

    results.append(_evaluate("xgboost", fitted_xgb, X_test, y_test, best_params))
    return results


def feature_importance(result: ModelResult, limit: int = 25) -> pd.DataFrame:
    """Return the largest model coefficients or tree importances."""

    pipeline = result.estimator
    if not isinstance(pipeline, Pipeline):
        return pd.DataFrame(columns=["feature", "importance"])
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = pd.DataFrame({"feature": names, "importance": values})
    importance["feature"] = importance["feature"].str.replace(
        r"^(numeric|categorical)__", "", regex=True
    )
    return importance.nlargest(limit, "importance").reset_index(drop=True)
