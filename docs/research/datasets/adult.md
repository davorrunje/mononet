# Datasheet — `adult` (Adult / Census Income)

*Gebru-style datasheet for the `adult` entry in [`datasets.yml`](../../../datasets.yml).
Drafted by the defendable-science `dataset` skill; classifications are author-signed
in the registry header.*

## Motivation

Predict whether a person's income exceeds $50K/yr from census attributes. Used in
`mononet`'s large-dataset screen as a **binary-classification** benchmark, with
income constrained **non-decreasing** in `education_num`, `hours_per_week`, and
`capital_gain`.

## Composition

- **Instances:** ~48842 people, one row per respondent.
- **Features:** demographic and employment attributes (age, education, marital
  status, occupation, hours-per-week, capital gain/loss, native country, …).
- **Target:** `ground_truth` — income >$50K (0/1).
- **Protected attributes present:** age, sex, race, native country.

## Collection process

Extracted from the 1994 US Census Bureau database by Barry Becker; deposited at
UCI. Records are **individual survey responses about real people**.

## Preprocessing

Committed to the repo (Git LFS) as gzipped train/test CSVs; regenerate with
`benchmarks/datasets/prepare/adult.py`.

## Uses & known limitations

Standard tabular fairness/monotonicity benchmark. Reflects 1994 US demographics
and known label/selection biases; not representative of any current population.

## Distribution & licensing

**CC-BY-4.0**. Source: <https://archive.ics.uci.edu/dataset/2/adult>.

## Sensitivity

**pii** — individual census records including protected attributes.

## Maintenance

Committed via Git LFS (Tier A); fixity pinned by SHA-256 in `datasets.yml`.
