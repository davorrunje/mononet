# Datasheet — `heart` (Cleveland Heart Disease)

*Gebru-style datasheet for the `heart` entry in [`datasets.yml`](../../../datasets.yml).
Drafted by the defendable-science `dataset` skill; classifications are author-signed
in the registry header.*

## Motivation

Predict the presence of heart disease in a patient from clinical measurements.
Used in `mononet` as a small **binary-classification** benchmark, with disease
risk constrained to be **non-decreasing** in `trestbps` (resting blood pressure)
and `chol` (serum cholesterol).

## Composition

- **Instances:** 303 patients (242 train / 61 test), one row per patient.
- **Features (13):** `age`, `sex`, `cp` (chest-pain type), `trestbps`, `chol`,
  `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
- **Target:** `ground_truth` — heart-disease presence (binarized 0/1 from the
  upstream 0–4 severity scale).
- The Cleveland subset of the UCI Heart Disease database (the one commonly used).

## Collection process

Collected at the Cleveland Clinic Foundation (Detrano et al., 1989) and deposited
at UCI. Records are **individual, de-identified patient data**: patient names and
identifiers were replaced with a dummy value upstream.

## Preprocessing

Redistributed as the paper's preprocessed train/test CSVs (Zenodo 7968969): the
14-attribute Cleveland subset, target binarized, fixed split.

## Uses

Monotone classification benchmark (Runje & Shankaranarayana 2023, Table 1). Not a
clinical tool; the sample is small and single-site.

## Distribution & licensing

Redistributed under **CC-BY-4.0** via Zenodo 7968969. Upstream:
<https://archive.ics.uci.edu/dataset/45/heart+disease>.

## Sensitivity

**pii** — individual-level health records. De-identified upstream, but the rows
are per-patient clinical measurements; treat as health-related personal data.

## Maintenance

Fixity pinned by SHA-256 in `datasets.yml` and `benchmarks/datasets/manifest.toml`.
Fetched at runtime (Tier B); not committed to the repo.
