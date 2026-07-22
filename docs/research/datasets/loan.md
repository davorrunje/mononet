# Datasheet — `loan` (Loan default)

*Gebru-style datasheet for the `loan` entry in [`datasets.yml`](../../../datasets.yml).
Drafted by the honest-scholar `dataset` skill; classifications are author-signed
in the registry header.*

## Motivation

Predict loan default from applicant and loan attributes. Used in `mononet` as the
largest **binary-classification** benchmark, with default risk constrained to be
**non-decreasing** in `feature_1`, `feature_4` and **non-increasing** in
`feature_0`, `feature_2`, `feature_3`.

## Composition

- **Instances:** 488909 loans (418697 train / 70212 test), one row per loan.
- **Features (28):** anonymized numeric features `feature_0..27`.
- **Target:** `ground_truth` — default indicator (0/1).
- Feature semantics are anonymized in the redistributed release; the monotone
  directions above are the paper's domain constraints on the (hidden) originals.

## Collection process

Consumer-lending records preprocessed for the CMNN paper. Rows are individual
loans; direct identifiers were removed and features anonymized upstream.

## Preprocessing

Redistributed as the paper's preprocessed train/test CSVs (Zenodo 7968969):
anonymized numeric features, fixed train/test split.

## Uses & known limitations

Monotone classification benchmark (Runje & Shankaranarayana 2023, Table 1).
Because features are anonymized, this entry supports reproduction of the paper's
results but not feature-level interpretation or any lending decision.

## Distribution & licensing

Redistributed under **CC-BY-4.0** via Zenodo 7968969.

## Sensitivity

**pii** — individual-level consumer-finance records. Anonymized upstream (no
direct identifiers), but treated as personal financial data, consistent with the
other credit datasets in this registry.

## Maintenance

Fixity pinned by SHA-256 in `datasets.yml` and `benchmarks/datasets/manifest.toml`.
Fetched at runtime (Tier B); not committed to the repo.
