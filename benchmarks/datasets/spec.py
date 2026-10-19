"""Per-dataset column descriptors (transcribed from airtai/monotonic-nn)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_SynthKind = Literal["additive", "teacher_relu", "teacher_elu", "lattice"]


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    """Generator descriptor for a synthetic (depth-probe) dataset.

    Carries the arguments forwarded verbatim to
    :func:`benchmarks.datasets.synthetic.synth_monotone`.

    :param kind: Target family.
    :param c: Complexity knob (teacher/lattice depth; ignored for additive).
    :param d: Input dimension.
    :param n_train: Train rows.
    :param n_test: Test rows.
    :param seed: RNG seed.
    """

    kind: _SynthKind
    c: int
    d: int
    n_train: int
    n_test: int
    seed: int


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Column layout descriptor for one benchmark dataset.

    :param name: Dataset key used in the registry and for filename resolution.
    :param task: Learning task type.
    :param target: Name of the target column in the CSV. Unused (empty) for
        generator-backed specs.
    :param mono_increasing: Feature names the target is non-decreasing in.
        For a generator-backed spec, these are the ``x{i}`` placeholder
        feature names, not real CSV column names.
    :param mono_decreasing: Feature names the target is non-increasing in.
    :param generator: If set, this dataset is produced by
        :func:`~benchmarks.datasets.synthetic.synth_monotone` instead of
        being loaded from CSV; :func:`~benchmarks.datasets.registry.load`
        ignores ``data_dir`` for such specs.
    """

    name: str
    task: Literal["binary_classification", "regression"]
    target: str
    mono_increasing: tuple[str, ...]
    mono_decreasing: tuple[str, ...]
    generator: GeneratorSpec | None = None


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

# Synthetic depth-probe datasets (#99): generator-backed, ignore `data_dir`.
# d=6, n_train=32000 (>20k -> large-batch band), n_test=4000 for all; c pinned per level
# (low=1, mid=2, high=4); one distinct seed per key.
_SYNTH_D = 6
_SYNTH_N_TRAIN = 32000
_SYNTH_N_TEST = 4000
_SYNTH_FEATURES = tuple(f"x{i}" for i in range(_SYNTH_D))
_SYNTH_LEVELS: tuple[tuple[str, int], ...] = (("low", 1), ("mid", 2), ("high", 4))
_SYNTH_FAMILIES: tuple[_SynthKind, ...] = (
    "additive",
    "teacher_relu",
    "teacher_elu",
    "lattice",
)

for _fam_idx, _family in enumerate(_SYNTH_FAMILIES):
    for _lvl_idx, (_level, _c) in enumerate(_SYNTH_LEVELS):
        _key = f"synth_{_family}_c{_level}"
        DATASETS_SPEC[_key] = DatasetSpec(
            _key,
            "regression",
            "",
            _SYNTH_FEATURES,
            (),
            generator=GeneratorSpec(
                kind=_family,
                c=_c,
                d=_SYNTH_D,
                n_train=_SYNTH_N_TRAIN,
                n_test=_SYNTH_N_TEST,
                seed=100 + _fam_idx * 10 + _lvl_idx,
            ),
        )
del _fam_idx, _family, _lvl_idx, _level, _c, _key
