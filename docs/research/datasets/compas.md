# Datasheet — `compas` (ProPublica COMPAS recidivism)

*Gebru-style datasheet for the `compas` entry in [`datasets.yml`](../../../datasets.yml).
Drafted by the defendable-science `dataset` skill; classifications are author-signed
in the registry header.*

## Motivation

Predict two-year recidivism for criminal defendants. Used in `mononet` as a
**binary-classification** benchmark, with risk constrained to be **non-decreasing**
in `priors_count`, `juv_fel_count`, `juv_misd_count`, and `juv_other_count`
(more prior/juvenile offenses ⇒ not-lower predicted risk).

## Composition

- **Instances:** 6172 defendants (4937 train / 1235 test), one row per person.
- **Features (13):** `priors_count`, `juv_fel_count`, `juv_misd_count`,
  `juv_other_count`, `age`, one-hot `race_0..5`, one-hot `sex_0..1`.
- **Target:** `ground_truth` — two-year recidivism (0/1).
- **Protected attributes are present:** race (6-way one-hot) and sex.

## Collection process

Assembled by ProPublica (Angwin et al., 2016) from Broward County, Florida public
records of COMPAS risk scores and subsequent arrests. Records are **individual
criminal-justice data on real people**.

## Preprocessing

Redistributed as the paper's preprocessed train/test CSVs (Zenodo 7968969): the
standard ProPublica filtering, race/sex one-hot encoded, fixed split.

## Uses & known limitations

Fairness/bias-sensitive. The COMPAS data is the canonical example of algorithmic
bias in criminal-justice risk scoring; arrest is a biased proxy for offending, and
the label reflects policing patterns, not ground-truth criminality. Use **only**
as a monotonicity/benchmark artifact (Runje & Shankaranarayana 2023, Table 1),
**never** to build or justify a real risk-assessment tool.

## Distribution & licensing

Redistributed under **CC-BY-4.0** via Zenodo 7968969. Upstream:
<https://github.com/propublica/compas-analysis>.

## Sensitivity

**pii** — individual criminal-justice records including protected attributes
(race, sex). Highest-sensitivity entry in the registry.

## Maintenance

Fixity pinned by SHA-256 in `datasets.yml` and `benchmarks/datasets/manifest.toml`.
Fetched at runtime (Tier B); not committed to the repo.
