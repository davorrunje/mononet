# Datasheet — `blog` (BlogFeedback)

*Gebru-style datasheet for the `blog` entry in [`datasets.yml`](../../../datasets.yml).
Drafted by the honest-scholar `dataset` skill; classifications are author-signed
in the registry header.*

## Motivation

Predict how many comments a blog post will receive in the next 24 hours from
features of the post and its recent comment history. Used in `mononet` as a large
**regression** benchmark, with the predicted comment count constrained to be
**non-decreasing** in a set of recent-activity features (`feature_50..53`,
`feature_55..59`).

## Composition

- **Instances:** 54270 blog-post observations (47302 train / 6968 test).
- **Features (276):** anonymized numeric features `feature_0..275` — basic,
  textual, and time-window activity statistics of the source page.
- **Target:** `ground_truth` — number of comments in the following 24 hours
  (continuous).

## Collection process

Crawled from Hungarian blog sites by Buza (2014); the raw HTML documents were
aggregated into per-post numeric features upstream. The unit of observation is a
blog post, not a person; no author identifiers are retained.

## Preprocessing

Redistributed as the paper's preprocessed train/test CSVs (Zenodo 7968969):
numeric features only, fixed train/test split.

## Uses

Monotone regression benchmark (Runje & Shankaranarayana 2023, Table 1).

## Distribution & licensing

Redistributed under **CC-BY-4.0** via Zenodo 7968969. Upstream:
<https://archive.ics.uci.edu/dataset/304/blogfeedback>.

## Sensitivity

**none** — aggregated per-post numeric statistics; no personal identifiers.

## Maintenance

Fixity pinned by SHA-256 in `datasets.yml` and `benchmarks/datasets/manifest.toml`.
Fetched at runtime (Tier B); not committed to the repo.
