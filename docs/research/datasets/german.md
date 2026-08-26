# Datasheet — `german` (Statlog German Credit)

*Gebru-style datasheet for the `german` entry in [`datasets.yml`](../../../datasets.yml).
Drafted by the defendable-science `dataset` skill; classifications are author-signed
in the registry header.*

## Motivation

Classify credit applicants as good or bad credit risks. Used in `mononet`'s
large-dataset screen as a **binary-classification** benchmark, with risk
constrained **non-decreasing** in `duration`, `credit_amount`, `installment_rate`
and **non-increasing** in `age`.

## Composition

- **Instances:** 1000 credit applicants, one row per person.
- **Features:** 20 attributes (account status, duration, credit history, purpose,
  credit amount, employment, age, housing, …).
- **Target:** `ground_truth` — good/bad credit risk (0/1).
- **Protected attributes present:** age; sex is encoded within personal-status.

## Collection process

Provided by Prof. Hans Hofmann (Universität Hamburg); part of the Statlog project.
Records are **individual credit applicants**.

## Preprocessing

Committed to the repo (Git LFS) as gzipped train/test CSVs; regenerate with
`benchmarks/datasets/prepare/german.py`.

## Uses & known limitations

Small, classic credit-scoring benchmark; a documented cost matrix (false-good vs
false-bad) exists upstream but is not used here. Known demographic biases.

## Distribution & licensing

**CC-BY-4.0**. Source:
<https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data>.

## Sensitivity

**pii** — individual-level credit records with demographic attributes.

## Maintenance

Committed via Git LFS (Tier A); fixity pinned by SHA-256 in `datasets.yml`.
