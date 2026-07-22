# Datasheet — `auto` (Auto MPG)

*Gebru-style datasheet for the `auto` entry in [`datasets.yml`](../../../datasets.yml).
Drafted by the honest-scholar `dataset` skill; classifications are author-signed
in the registry header.*

## Motivation

Predict a car's city-cycle fuel consumption (miles-per-gallon) from its physical
and engine attributes. Used in `mononet` as a small **regression** benchmark for
the constrained-monotonic construction, with fuel economy constrained to be
**non-increasing** in `Weight`, `Displacement`, and `Horsepower` (heavier /
larger-engined cars use more fuel).

## Composition

- **Instances:** 392 cars (314 train / 78 test), one row per vehicle.
- **Features (7):** `Cylinders`, `Displacement`, `Horsepower`, `Weight`,
  `Acceleration`, `Model Year`, `Origin`.
- **Target:** `ground_truth` — miles-per-gallon (continuous).
- **Missing values:** the six upstream rows with a missing `Horsepower` are
  dropped in the preprocessed release (398 → 392).

## Collection process

Compiled by the StatLib library (CMU) from 1983 American Statistical Association
Exposition data; introduced by Quinlan (1993). No individuals are involved — the
unit of observation is a car model.

## Preprocessing

Redistributed as the paper's preprocessed train/test CSVs (Zenodo 7968969):
NA-`Horsepower` rows removed, a fixed train/test split, numeric encoding.

## Uses

Monotone regression benchmark (Runje & Shankaranarayana 2023, Table 1). Not
suitable for any inference about people.

## Distribution & licensing

Redistributed under **CC-BY-4.0** via Zenodo 7968969. Upstream:
<https://archive.ics.uci.edu/dataset/9/auto+mpg>.

## Sensitivity

**none** — vehicle specifications only; no personal or confidential data.

## Maintenance

Fixity pinned by SHA-256 in `datasets.yml` and `benchmarks/datasets/manifest.toml`.
Fetched at runtime (Tier B); not committed to the repo.
