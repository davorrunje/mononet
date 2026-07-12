"""Per-dataset column descriptors (transcribed from airtai/monotonic-nn)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Column layout descriptor for one benchmark dataset.

    :param name: Dataset key used in the registry and for filename resolution.
    :param task: Learning task type.
    :param target: Name of the target column in the CSV.
    :param mono_increasing: Feature names the target is non-decreasing in.
    :param mono_decreasing: Feature names the target is non-increasing in.
    """

    name: str
    task: Literal["binary_classification", "regression"]
    target: str
    mono_increasing: tuple[str, ...]
    mono_decreasing: tuple[str, ...]


DATASETS_SPEC: dict[str, DatasetSpec] = {
    "auto": DatasetSpec(
        "auto",
        "regression",
        "ground_truth",
        (),
        ("Weight", "Displacement", "Horsepower"),
    ),
    "heart": DatasetSpec(
        "heart",
        "binary_classification",
        "ground_truth",
        ("trestbps", "chol"),
        (),
    ),
    "compas": DatasetSpec(
        "compas",
        "binary_classification",
        "ground_truth",
        ("priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"),
        (),
    ),
    "loan": DatasetSpec(
        "loan",
        "binary_classification",
        "ground_truth",
        ("feature_1", "feature_4"),
        ("feature_0", "feature_2", "feature_3"),
    ),
    "blog": DatasetSpec(
        "blog",
        "regression",
        "ground_truth",
        tuple(f"feature_{i}" for i in (50, 51, 52, 53, 55, 56, 57, 58, 59)),
        (),
    ),
    "adult": DatasetSpec(
        "adult",
        "binary_classification",
        "ground_truth",
        ("education_num", "hours_per_week", "capital_gain"),
        (),
    ),
    "taiwan": DatasetSpec(
        "taiwan",
        "binary_classification",
        "ground_truth",
        ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
        (
            "LIMIT_BAL",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ),
    ),
    "polish": DatasetSpec(
        "polish",
        "binary_classification",
        "ground_truth",
        ("Attr2",),
        ("Attr1", "Attr4", "Attr17", "Attr23", "Attr35"),
    ),
    "german": DatasetSpec(
        "german",
        "binary_classification",
        "ground_truth",
        ("duration", "credit_amount", "installment_rate"),
        ("age",),
    ),
    "lc": DatasetSpec(
        "lc",
        "binary_classification",
        "ground_truth",
        ("dti_n",),
        ("fico_n", "revenue"),
    ),
}
