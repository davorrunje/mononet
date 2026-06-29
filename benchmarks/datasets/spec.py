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
}
