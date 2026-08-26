# Datasheet — `taiwan` (Default of Credit Card Clients)

*Gebru-style datasheet for the `taiwan` entry in [`datasets.yml`](../../../datasets.yml).
Drafted by the defendable-science `dataset` skill; classifications are author-signed
in the registry header.*

## Motivation

Predict credit-card default from client and billing attributes. Used in
`mononet`'s large-dataset screen as a **binary-classification** benchmark, with
default risk constrained **non-decreasing** in the repayment-status features
`PAY_0`, `PAY_2..PAY_6` and **non-increasing** in `LIMIT_BAL` and the payment
amounts `PAY_AMT1..PAY_AMT6`.

## Composition

- **Instances:** 30000 credit-card clients, one row per client.
- **Features:** 23 attributes (credit limit, sex, education, marriage, age, six
  months of repayment status, bill statements, and payment amounts).
- **Target:** `ground_truth` — default next month (0/1).
- **Protected attributes present:** sex, age, education, marital status.

## Collection process

Taiwanese credit-card clients, October 2005 (Yeh & Lien, 2009); deposited at UCI.
Records are **individual clients**.

## Preprocessing

Committed to the repo (Git LFS) as gzipped train/test CSVs; regenerate with
`benchmarks/datasets/prepare/taiwan.py`.

## Uses & known limitations

Standard credit-default benchmark. Single-market, single-period; known
demographic biases.

## Distribution & licensing

**CC-BY-4.0**. Source:
<https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip>.

## Sensitivity

**pii** — individual-level financial records with demographic attributes.

## Maintenance

Committed via Git LFS (Tier A); fixity pinned by SHA-256 in `datasets.yml`.
