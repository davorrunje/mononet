# Methodology references

This document collects verified primary sources for the experimental-methodology
practices used in this project. Its purpose is twofold: to supply citable
references for the methodology section of the paper (whose flagship result is a
*null* — "network depth does not improve accuracy for constrained monotone
networks"), and to ground the templates of our hypothesis-recording system (a
pre-experiment **strategy** doc and a post-experiment **findings** doc). A null
result carries a higher evidential burden than a positive one: it must be
pre-committed, adequately powered, and reported so that "no effect" is
distinguishable from "no data." Every citation below was checked against its
publisher of record; none is marked UNVERIFIED.

## 1. Pre-registration, confirmatory vs. exploratory analysis, HARKing, forking paths, p-hacking

- **Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018).**
  *The preregistration revolution.* PNAS 115(11): 2600–2606.
  DOI: [10.1073/pnas.1708274114](https://doi.org/10.1073/pnas.1708274114).
  Key takeaway: Committing hypotheses and the analysis plan to a time-stamped
  record before seeing outcomes preserves the distinction between confirmatory
  (diagnostic) and exploratory (generative) claims, and is the most direct
  defense of a stated error rate.
  Template use: The **strategy** doc *is* the preregistration — it fixes the
  hypothesis, primary metric, and decision rule before any run; the **findings**
  doc must label every analysis not in the strategy doc as exploratory.

- **Kerr, N. L. (1998).** *HARKing: Hypothesizing After the Results are Known.*
  Personality and Social Psychology Review 2(3): 196–217.
  DOI: [10.1207/s15327957pspr0203_4](https://doi.org/10.1207/s15327957pspr0203_4).
  Key takeaway: Presenting a post-hoc hypothesis as if it had been a priori
  inflates false positives and misrepresents the evidence; the fix is a durable
  a-priori record of what was predicted.
  Template use: The strategy doc timestamps the a-priori hypothesis so the
  findings doc cannot retrofit the story to the data.

- **Gelman, A., & Loken, E. (2014).** *The statistical crisis in science*
  ("the garden of forking paths"). American Scientist 102(6): 460–465.
  DOI: [10.1511/2014.111.460](https://doi.org/10.1511/2014.111.460)
  (working-paper version: [Columbia PDF](https://sites.stat.columbia.edu/gelman/research/unpublished/p_hacking.pdf)).
  Key takeaway: Data-contingent analysis choices create an implicit multiple-
  comparisons problem even without conscious p-hacking and even when the
  hypothesis was posited in advance — the many analyses you *would* have run
  count against you.
  Template use: The strategy doc should pre-specify preprocessing, metric, and
  the exact comparison so the analysis path is not chosen after seeing data.

- **Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011).** *False-Positive
  Psychology: Undisclosed Flexibility in Data Collection and Analysis Allows
  Presenting Anything as Significant.* Psychological Science 22(11): 1359–1366.
  DOI: [10.1177/0956797611417632](https://doi.org/10.1177/0956797611417632).
  Key takeaway: "Researcher degrees of freedom" (optional stopping, selective
  reporting of conditions/metrics, flexible exclusions) push actual false-
  positive rates far above the nominal 0.05; disclosure requirements curb it.
  Template use: The findings doc should include the disclosure checklist —
  all conditions run, all metrics collected, stopping rule, exclusions.

## 2. Strong inference and multiple working hypotheses

- **Platt, J. R. (1964).** *Strong Inference.* Science 146(3642): 347–353
  (republished 1965). DOI: [10.1126/science.146.3642.347](https://doi.org/10.1126/science.146.3642.347).
  Key takeaway: Progress is fastest when each experiment is designed to
  *exclude* hypotheses via explicit, disconfirming tests, iterated as a
  conditional tree rather than accumulating confirmations of one favored idea.
  Template use: The strategy doc should frame each experiment as a decisive
  test that can rule a candidate explanation *out*, not merely support the
  preferred one.

- **Chamberlin, T. C. (1890 / 1965).** *The Method of Multiple Working
  Hypotheses.* Science (old series) 15: 92–96; reprinted Science 148(3671):
  754–759 (1965). DOI: [10.1126/science.148.3671.754](https://doi.org/10.1126/science.148.3671.754).
  Key takeaway: Holding several rival explanations simultaneously guards against
  the parental attachment to a single hypothesis that biases design and
  interpretation.
  Template use: The strategy doc should enumerate competing explanations for the
  depth-null (e.g., true no-effect vs. under-powered test vs. optimization
  failure) so the findings doc can adjudicate among them rather than confirm one.

## 3. Severity / error-statistical testing

- **Mayo, D. G. (2018).** *Statistical Inference as Severe Testing: How to Get
  Beyond the Statistics Wars.* Cambridge University Press.
  ISBN 978-1-107-05413-4.
  [Cambridge Core](https://www.cambridge.org/core/books/statistical-inference-as-severe-testing/D9DF409EF6D65EAA2C4C0C4D5A78F35A).
  Key takeaway: A claim is warranted only to the degree it has passed a *severe*
  test — one that would very probably have found the claim false had it been
  false; passing a test that could not have failed carries no evidential weight.
  Template use: Especially load-bearing for a null — the findings doc must argue
  the "no depth effect" claim survived a test with high probability of detecting
  a depth effect had one existed (i.e., adequate power / severity), not merely
  that p > 0.05.

## 4. Statistical reporting for ML / RL

- **Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G.
  (2021).** *Deep Reinforcement Learning at the Edge of the Statistical
  Precipice.* NeurIPS 2021 (Outstanding Paper).
  arXiv: [2108.13264](https://arxiv.org/abs/2108.13264);
  library [`rliable`](https://github.com/google-research/rliable).
  Key takeaway: Point estimates over a handful of seeds are unreliable; report
  the interquartile mean (IQM), stratified bootstrap confidence intervals, and
  performance profiles to convey uncertainty and robustness to outliers.
  Template use: The strategy doc should specify seeds, the IQM aggregate, and CI
  method up front; the findings doc reports IQM + stratified-bootstrap CIs and
  performance profiles rather than a single mean accuracy — critical when the
  claim is that two conditions (shallow vs. deep) do *not* differ.

## 5. Equivalence testing and power / minimum-detectable-effect for null results

- **Lakens, D. (2017).** *Equivalence Tests: A Practical Primer for t Tests,
  Correlations, and Meta-Analyses.* Social Psychological and Personality Science
  8(4): 355–362. DOI: [10.1177/1948550617697177](https://doi.org/10.1177/1948550617697177).
  Key takeaway: The two one-sided tests (TOST) procedure lets you *statistically
  reject* the presence of an effect large enough to matter, converting "we found
  nothing" into a positive claim of equivalence within a pre-set bound.
  Template use: The strategy doc must set the smallest effect size of interest
  (the equivalence bound / minimum-detectable accuracy gap); the findings doc
  runs TOST against it.

- **Lakens, D., Scheel, A. M., & Isager, P. M. (2018).** *Equivalence Testing
  for Psychological Research: A Tutorial.* Advances in Methods and Practices in
  Psychological Science 1(2): 259–269.
  DOI: [10.1177/2515245918770963](https://doi.org/10.1177/2515245918770963).
  Key takeaway: Step-by-step guidance on choosing equivalence bounds and running
  TOST, with power analysis for the bounds.
  Template use: Procedural reference for implementing the strategy doc's
  equivalence test and its power/MDE justification.

- **Altman, D. G., & Bland, J. M. (1995).** *Statistics notes: Absence of
  evidence is not evidence of absence.* BMJ 311(7003): 485.
  DOI: [10.1136/bmj.311.7003.485](https://doi.org/10.1136/bmj.311.7003.485).
  Key takeaway: A non-significant result with inadequate power shows only that
  no effect was *detected*, not that none exists; a null claim requires a study
  powered to detect the effect it denies.
  Template use: The strategy doc must include an a-priori power / minimum-
  detectable-effect analysis so the depth-null is not merely an under-powered
  non-result.

## 6. ML reproducibility standards

- **Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A.,
  d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021).** *Improving Reproducibility
  in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility
  Program).* JMLR 22(164): 1–20.
  [JMLR](https://www.jmlr.org/papers/v22/20-303.html);
  arXiv: [2003.12206](https://arxiv.org/abs/2003.12206).
  Key takeaway: Documents the ML reproducibility checklist, code-submission
  policy, and reproducibility challenge — concrete artifact/reporting standards
  now standard at major ML venues.
  Template use: The strategy doc adopts the reproducibility checklist as an
  intake gate (code, data, compute, hyperparameters); the findings doc attaches
  the completed checklist.

- **Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson,
  B., Spitzer, E., Raji, I. D., & Gebru, T. (2019).** *Model Cards for Model
  Reporting.* FAT* '19: 220–229.
  DOI: [10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596);
  arXiv: [1810.03993](https://arxiv.org/abs/1810.03993).
  Key takeaway: A short structured record of a model's intended use, training
  and evaluation data, metrics, and performance across conditions promotes
  transparent, disaggregated reporting.
  Template use: The findings doc can carry a model-card-style summary of each
  trained configuration (shallow / deep) and its evaluation conditions.

- **Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H.,
  Daumé III, H., & Crawford, K. (2021).** *Datasheets for Datasets.*
  Communications of the ACM 64(12): 86–92.
  DOI: [10.1145/3458723](https://doi.org/10.1145/3458723).
  Key takeaway: Standardized dataset documentation (motivation, composition,
  collection, preprocessing, uses, distribution, maintenance) makes provenance
  and fitness-for-purpose explicit.
  Template use: The strategy doc records a datasheet-style description of each
  benchmark dataset so dataset choice cannot silently confound the depth
  comparison.

## 7. FAIR data principles

- **Wilkinson, M. D., Dumontier, M., Aalbersberg, Ij. J., et al. (2016).**
  *The FAIR Guiding Principles for scientific data management and stewardship.*
  Scientific Data 3: 160018.
  DOI: [10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18).
  Key takeaway: Data and metadata should be Findable, Accessible, Interoperable,
  and Reusable — by machines as well as humans — via persistent identifiers and
  rich, standardized metadata.
  Template use: Both docs reference FAIR-compliant artifacts — persistent IDs for
  datasets, configs, and result tables — so a null result is independently
  re-checkable.

## 8. Experiment-tracking practices

The value here is the *practices* these tools encode, not any specific tool.
Config provenance and run tracking make the strategy doc's pre-committed
settings auditable against what was actually executed, and let the findings doc
tie every reported number to a specific run, code state, and hyperparameter set.
**Sacred** (Greff, K., Klein, A., Chovanec, M., Hutter, F., & Schmidhuber, J.,
2017, *The Sacred Infrastructure for Computational Research*, SciPy 2017,
DOI: [10.25080/shinma-7f4c6e7-008](https://doi.org/10.25080/shinma-7f4c6e7-008))
formalizes automatic capture of configuration, seeds, and outputs per run.
Complementary systems encode the same principles:
[MLflow](https://mlflow.org/) (run/param/metric/artifact tracking),
[Weights & Biases](https://wandb.ai/) (experiment logging and dashboards),
[Hydra](https://hydra.cc/) (composable, overridable, logged configuration), and
[DVC](https://dvc.org/) (versioning of data and pipelines alongside code).
Together they operationalize the reproducibility checklist (§6) and FAIR (§7):
every seed, config, and metric is a versioned, retrievable artifact.

## How this maps to the templates

**Pre-experiment strategy doc** (write before any run):
- A-priori hypothesis, primary metric, and decision rule — timestamped
  (Nosek 2018; Kerr 1998).
- Pre-specified preprocessing and the exact comparison, closing off the garden
  of forking paths (Gelman & Loken 2014; Simmons et al. 2011).
- Enumerated rival hypotheses framed as decisive/disconfirming tests
  (Platt 1964; Chamberlin 1890/1965).
- **For the null:** equivalence bound (smallest effect of interest) plus an
  a-priori power / minimum-detectable-effect analysis, so the test is severe
  (Lakens 2017/2018; Altman & Bland 1995; Mayo 2018).
- Reporting plan: seeds, IQM + stratified-bootstrap CIs, performance profiles
  (Agarwal et al. 2021); reproducibility checklist intake and datasheet for each
  dataset (Pineau et al. 2021; Gebru et al. 2021).
- FAIR persistent IDs and experiment-tracking config provenance registered up
  front (Wilkinson et al. 2016; Sacred/MLflow/W&B/Hydra/DVC, §8).

**Post-experiment findings doc** (write after runs):
- Confirmatory results reported against the pre-registered plan; every extra
  analysis explicitly labeled exploratory (Nosek 2018).
- Full disclosure checklist — all conditions, metrics, stopping rule, exclusions
  (Simmons et al. 2011).
- IQM with stratified-bootstrap CIs and performance profiles, not a bare mean
  (Agarwal et al. 2021).
- **For the null:** TOST equivalence result against the pre-set bound plus the
  realized power, arguing the test was severe rather than merely non-significant
  (Lakens 2017/2018; Altman & Bland 1995; Mayo 2018).
- Adjudication among the pre-listed rival hypotheses (Platt 1964; Chamberlin).
- Completed reproducibility checklist, model-card summaries per configuration,
  and links to FAIR-identified, tracked run artifacts (Pineau et al. 2021;
  Mitchell et al. 2019; Wilkinson et al. 2016; §8).
