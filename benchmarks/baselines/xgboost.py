"""XGBoost baseline model runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xgboost import XGBClassifier, XGBRegressor

if TYPE_CHECKING:
    from benchmarks._common.bundle import DatasetBundle


def run_xgboost(bundle: DatasetBundle, *, seed: int = 0) -> dict[str, float]:
    """Run XGBoost baseline on the given dataset bundle.

    :param bundle: The dataset bundle containing train/test splits and task type.
    :param seed: Random seed for reproducibility.
    :returns: Dictionary of metrics. For regression: ``{"mse": ..., "rmse": ...}``.
              For binary classification: ``{"accuracy": ...}``.
    :raises ValueError: If task type is neither "regression" nor "binary_classification".
    """
    if bundle.task == "regression":
        model = XGBRegressor(
            random_state=seed,
            verbosity=0,
        )
        model.fit(bundle.X_train, bundle.y_train)
        y_pred = model.predict(bundle.X_test)

        # Compute metrics
        mse = float(((bundle.y_test - y_pred) ** 2).mean())
        rmse = float(mse ** 0.5)

        return {"mse": mse, "rmse": rmse}

    elif bundle.task == "binary_classification":
        model = XGBClassifier(
            random_state=seed,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(bundle.X_train, bundle.y_train)
        y_pred = model.predict(bundle.X_test)

        # Compute metrics
        accuracy = float((bundle.y_test == y_pred).mean())

        return {"accuracy": accuracy}

    else:
        raise ValueError(
            f"Unsupported task type: {bundle.task}. "
            f"Expected 'regression' or 'binary_classification'."
        )
