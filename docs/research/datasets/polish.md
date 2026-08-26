# Datasheet — `polish` (Polish Companies Bankruptcy)

*Gebru-style datasheet for the `polish` entry in [`datasets.yml`](../../../datasets.yml).
Drafted by the defendable-science `dataset` skill; classifications are author-signed
in the registry header.*

## Motivation

Predict corporate bankruptcy from financial ratios. Used in `mononet`'s
large-dataset screen as a **binary-classification** benchmark, with bankruptcy
risk constrained **non-decreasing** in `Attr2` and **non-increasing** in `Attr1`,
`Attr4`, `Attr17`, `Attr23`, `Attr35`.

## Composition

- **Instances:** ~10000 company-year observations, one row per company-year.
- **Features:** 64 econometric/financial ratios (`Attr1..Attr64`).
- **Target:** `ground_truth` — bankruptcy within the forecast horizon (0/1).
- The unit of observation is a **company**, not a person.

## Collection process

Polish companies analyzed 2000–2013 (emerging-markets data from EMIS); deposited
at UCI by Zięba et al. (2016).

## Preprocessing

Committed to the repo (Git LFS) as gzipped train/test CSVs; regenerate with
`benchmarks/datasets/prepare/polish.py`.

## Uses & known limitations

Bankruptcy-prediction benchmark. Strong class imbalance and many missing ratios
upstream; imputation choices affect results.

## Distribution & licensing

**CC-BY-4.0**. Source:
<https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip>.

## Sensitivity

**none** — company-level financial data; no personal data.

## Maintenance

Committed via Git LFS (Tier A); fixity pinned by SHA-256 in `datasets.yml`.
