# Datasheet — `lc` (Lending Club)

*Gebru-style datasheet for the `lc` entry in [`datasets.yml`](../../../datasets.yml).
Drafted by the honest-scholar `dataset` skill; classifications are author-signed
in the registry header.*

## Motivation

Predict loan outcome from Lending Club consumer-loan attributes. Used as the
largest entry in `mononet`'s large-dataset screen (**binary classification**),
with risk constrained **non-decreasing** in `dti_n` (debt-to-income) and
**non-increasing** in `fico_n` (FICO score) and `revenue`.

## Composition

- **Instances:** large (hundreds of thousands of loans), one row per loan.
- **Features:** borrower and loan attributes (FICO band, DTI, income/revenue, …),
  numeric-encoded.
- **Target:** `ground_truth` — loan default/charge-off indicator (0/1).

## Collection process

Public Lending Club marketplace loan data, redistributed via a Zenodo record.
Rows are individual loans; direct identifiers are not present.

## Preprocessing

Committed to the repo (Git LFS) as gzipped train/test CSVs; regenerate with
`benchmarks/datasets/prepare/lc.py`.

## Uses & known limitations

Large-scale credit-scoring benchmark. Subject to origination/selection bias
(only funded loans) and temporal drift; use as a benchmark artifact only.

## Distribution & licensing

**CC-BY-4.0**. Source: <https://zenodo.org/records/11295916>.

## Sensitivity

**pii** — individual-level consumer-finance records.

## Maintenance

Committed via Git LFS (Tier A); fixity pinned by SHA-256 in `datasets.yml`.
