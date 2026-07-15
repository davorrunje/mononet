# Hypothesis-generation references

This document collects verified primary sources on the *methodology of
generating and prioritizing new scientific hypotheses* — the logic by which a
researcher moves from a corpus of already-tested claims and their findings to
the next set of candidate hypotheses worth testing. It is the companion to
[methodology-references.md](methodology-references.md), which covers the
*confirmatory* half of the loop (pre-registration, strong inference, severity,
`rliable` reporting, equivalence testing, reproducibility); this file
deliberately does **not** repeat those and instead grounds a
"hypothesis-exploration" research skill that reads the existing hypothesis
record and proposes new candidates for the (separate, pre-registered) testing
skill to adjudicate. Every citation below was checked against its publisher of
record, arXiv, or the ACL Anthology; none is marked UNVERIFIED.

## 1. Abduction / inference to the best explanation

Abduction is the logic of hypothesis *generation*: from a surprising
observation, infer the hypothesis that, if true, would render it a matter of
course. It is the generative counterpart to the confirmatory tools in the
companion doc.

- **Peirce, C. S. (1931–1958).** *Collected Papers of Charles Sanders Peirce*
  (eds. C. Hartshorne, P. Weiss, A. Burks), Harvard University Press. The
  abduction–deduction–induction cycle is developed chiefly in Vol. 5
  (*Pragmatism and Pragmaticism*). Peirce coined "abduction" for the
  non-deductive inference that *proposes* an explanatory hypothesis, distinct
  from deduction (which draws its consequences) and induction (which tests it).
  Key takeaway: Discovery is a three-phase cycle — abduce a candidate
  explanation, deduce its testable consequences, induce a verdict from data —
  and only abduction introduces new ideas.
  Skill use: Frames the skill's core loop; the ingested corpus is the
  "surprising observations," and each emitted candidate is an abduced
  explanation carrying deducible, testable consequences for the testing skill.

- **Douven, I. (2021, rev.).** *Abduction.* Stanford Encyclopedia of Philosophy.
  Stable URL: <https://plato.stanford.edu/entries/abduction/>
  (supplement: <https://plato.stanford.edu/entries/abduction/peirce.html>).
  Key takeaway: A scholarly, citable synthesis distinguishing Peirce's
  *generative* sense of abduction from the modern *justificatory* sense
  (inference to the best explanation), and cataloguing the explanatory virtues
  (simplicity, scope, coherence) used to rank rival explanations.
  Skill use: Supplies the ranking criteria for the "surviving rival
  explanations" the skill must weigh when it proposes which explanation to test
  next.

- **Lipton, P. (2004).** *Inference to the Best Explanation*, 2nd edition.
  Routledge (International Library of Philosophy). ISBN 0-415-24203-7.
  Key takeaway: We do not infer the *likeliest* explanation but the *loveliest*
  — the one that would, if correct, provide the most understanding — while
  guarding against inferring the best of a bad lot; the second edition adds the
  reconciliation with Bayesianism.
  Skill use: Justifies generating an explicit *slate* of rival explanations
  (not one) and warns the skill to flag when the best available candidate is
  still weak ("best of a bad lot"), so it does not over-promote a poor idea.

## 2. Exploratory vs. confirmatory research and EDA as a hypothesis generator

- **Tukey, J. W. (1977).** *Exploratory Data Analysis.* Addison-Wesley, Reading,
  MA. ISBN 0-201-07616-0.
  Key takeaway: EDA is detective work whose job is to *suggest* hypotheses and
  reveal structure ("the greatest value of a picture is when it forces us to
  notice what we never expected to see"), a role sharply separate from
  confirmatory data analysis that tests them.
  Skill use: Legitimizes mining the corpus's *exploratory observations* for
  patterns and candidate effects — while requiring that anything so generated be
  emitted as an open hypothesis for pre-registered testing, never reported as
  confirmed.

- **Wagenmakers, E.-J., Wetzels, R., Borsboom, D., van der Maas, H. L. J., &
  Kievit, R. A. (2012).** *An Agenda for Purely Confirmatory Research.*
  Perspectives on Psychological Science 7(6): 632–638.
  DOI: [10.1177/1745691612463078](https://doi.org/10.1177/1745691612463078).
  Key takeaway: Exploration and confirmation are both essential but must be kept
  strictly apart; exploratory findings are hypothesis-*generating* and become
  evidence only after a fresh, pre-committed confirmatory test.
  Skill use: Fixes the division of labor — this skill owns exploration and
  hand-off, and is explicitly barred from confirming its own outputs; that role
  belongs to the pre-registered testing skill.

## 3. Bayesian optimal experimental design / expected information gain

The question "which hypothesis or experiment do we run *next*?" is exactly the
one Bayesian optimal experimental design (BOED) answers, by choosing the design
that maximizes expected information gain (EIG) about the quantity of interest.

- **Lindley, D. V. (1956).** *On a Measure of the Information Provided by an
  Experiment.* Annals of Mathematical Statistics 27(4): 986–1005.
  DOI: [10.1214/aoms/1177728069](https://doi.org/10.1214/aoms/1177728069).
  Key takeaway: The value of an experiment is the expected reduction in Shannon
  entropy over the parameter of interest (prior vs. posterior) — the founding
  definition of expected information gain.
  Skill use: The formal backbone for prioritizing candidates: prefer the
  hypothesis-test whose result would most reduce uncertainty about the open
  question.

- **Chaloner, K., & Verdinelli, I. (1995).** *Bayesian Experimental Design: A
  Review.* Statistical Science 10(3): 273–304.
  DOI: [10.1214/ss/1177009939](https://doi.org/10.1214/ss/1177009939).
  Key takeaway: Casts experimental design as decision theory — choose the design
  maximizing expected utility, with EIG as the canonical utility — unifying
  linear and nonlinear design problems.
  Skill use: Grounds a *utility-based* ranking that combines information gain
  with cost and relevance, rather than information gain alone.

- **Foster, A., Jankowiak, M., Bingham, E., Horsfall, P., Teh, Y. W., Rainforth,
  T., & Goodman, N. (2019).** *Variational Bayesian Optimal Experimental
  Design.* NeurIPS 2019. arXiv: [1903.05480](https://arxiv.org/abs/1903.05480).
  Key takeaway: EIG is intractable in general; amortized/variational lower
  bounds make it estimable and let design and inference be optimized jointly by
  stochastic gradient ascent.
  Skill use: Signals that EIG scores for candidate experiments can be
  *approximated* cheaply — the skill can rank by an estimated EIG proxy rather
  than an exact (unavailable) value.

- **Rainforth, T., Foster, A., Ivanova, D. R., & Bickford Smith, F. (2024).**
  *Modern Bayesian Experimental Design.* Statistical Science 39(1): 100–114.
  DOI: [10.1214/23-STS915](https://doi.org/10.1214/23-STS915);
  arXiv: [2302.14545](https://arxiv.org/abs/2302.14545).
  Key takeaway: A current review of scalable, gradient-based, and adaptive
  (sequential) BOED, covering how each new result reshapes the design of the
  next experiment.
  Skill use: Directly models the corpus→next-hypothesis loop as *sequential*
  BOED — each recorded finding updates the posterior that determines which
  candidate is now most informative to test.

## 4. Anomaly-driven discovery

- **Kuhn, T. S. (1962; 2nd ed. 1970; 50th-anniversary ed. 2012).** *The
  Structure of Scientific Revolutions.* University of Chicago Press
  (Ch. VI, "Anomaly and the Emergence of Scientific Discoveries").
  ISBN 978-0-226-45812-0 (2012 ed.).
  Key takeaway: Discovery begins with the recognition of *anomaly* — an
  observation the reigning paradigm cannot accommodate; persistent anomalies
  drive the search for new theory.
  Skill use: Motivates an "anomaly" generation move — surface corpus findings
  that contradict the prevailing model/expectation (e.g., the project's
  depth-null) and turn each into a candidate hypothesis about *why* the
  expectation failed.

## 5. Analogical reasoning in scientific discovery

- **Gentner, D. (1983).** *Structure-Mapping: A Theoretical Framework for
  Analogy.* Cognitive Science 7(2): 155–170.
  DOI: [10.1207/s15516709cog0702_3](https://doi.org/10.1207/s15516709cog0702_3).
  Key takeaway: Analogy is a mapping of *relational structure* (not surface
  attributes) from a base domain to a target; the "systematicity principle"
  favors mapping deep, interconnected relations — the mechanism behind many
  scientific analogies (e.g., heat-flow ↔ water-flow).
  Skill use: Formalizes a "generalization/transfer" move — carry a confirmed
  relational finding in one setting (dataset, backend, architecture) to a new
  target where the same relation might hold, and emit that transfer as a
  candidate hypothesis.
  *(Complementary: Holyoak & Thagard's multiconstraint theory of analogy
  extends this with similarity/structure/purpose constraints; cited here only as
  a pointer, not verified in detail.)*

## 6. Boundary conditions / scope of theories as a source of new questions

- **Busse, C., Kach, A. P., & Wagner, S. M. (2017).** *Boundary Conditions: What
  They Are, How to Explore Them, Why We Need Them, and When to Consider Them.*
  Organizational Research Methods 20(4): 574–609.
  DOI: [10.1177/1094428116641191](https://doi.org/10.1177/1094428116641191).
  Key takeaway: Boundary conditions are the "who/where/when" limits of a
  theory's validity; making them explicit both sharpens a theory and generates
  new research questions at its edges.
  Skill use: Grounds a "boundary/scope" generation move — probe *where* a
  confirmed finding stops holding (which datasets, scales, regimes) and emit each
  suspected limit as a candidate hypothesis to test.

## 7. Computational and automated scientific discovery

Historically robust foundations, then a fast-moving and easy-to-misquote recent
LLM literature. The 2024–2025 items below are **verified against arXiv / the ACL
Anthology** but should still be treated as *emerging* evidence, not settled
method.

- **Langley, P., Simon, H. A., Bradshaw, G. L., & Żytkow, J. M. (1987).**
  *Scientific Discovery: Computational Explorations of the Creative Processes.*
  MIT Press. ISBN 0-262-62052-9 (paperback) / 0-262-12116-6 (hardcover).
  Key takeaway: The BACON family of programs rediscovered empirical laws (e.g.,
  Kepler's third law) by heuristic search for regularities in data — an early
  demonstration that hypothesis generation is a searchable, mechanizable
  process.
  Skill use: Precedent that pattern-driven law induction over recorded data can
  be automated; the skill's generation moves are a modern, corpus-driven analogue.

- **King, R. D., Rowland, J., Oliver, S. G., et al. (2009).** *The Automation of
  Science.* Science 324(5923): 85–89.
  DOI: [10.1126/science.1165620](https://doi.org/10.1126/science.1165620).
  Key takeaway: The "Robot Scientist" Adam autonomously formed functional-genomics
  hypotheses about yeast, designed and ran experiments to test them, and
  interpreted the results — a closed generate→test→interpret loop.
  Skill use: The reference architecture for the two-skill system: generation
  (this skill) coupled to automated, pre-committed testing — with the human/other
  skill closing the loop.

- **Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024).**
  *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.*
  arXiv: [2408.06292](https://arxiv.org/abs/2408.06292) (submitted 12 Aug 2024).
  Key takeaway: An LLM pipeline that generates ML research ideas, implements and
  runs experiments, and writes up papers with an automated reviewer — an
  end-to-end (if uneven) demonstration of LLM-driven discovery.
  Skill use: Closest analogue to the intended skill; its idea-generation +
  novelty/feasibility filtering stage is the pattern to emulate, while its known
  reliability limits argue for the strict generation/confirmation split.

- **Yang, Z., Du, X., Li, J., Zheng, J., Poria, S., & Cambria, E. (2024).**
  *Large Language Models for Automated Open-domain Scientific Hypotheses
  Discovery.* Findings of the ACL 2024: 13545–13565. ACL Anthology:
  [2024.findings-acl.804](https://aclanthology.org/2024.findings-acl.804/);
  arXiv: [2309.02726](https://arxiv.org/abs/2309.02726).
  Key takeaway: Frames LLM hypothesis generation as *open-domain hypothetical
  induction* — from raw observations to novel, valid hypotheses without
  hand-picked premises — with a multi-module (MOOSE) generate-and-critique
  approach.
  Skill use: A concrete recipe for the induction step: generate candidate
  hypotheses from observations, then self-critique for novelty and validity
  before hand-off.

- **Gottweis, J., et al. (2025).** *Towards an AI Co-scientist.*
  arXiv: [2502.18864](https://arxiv.org/abs/2502.18864) (Feb 2025).
  Key takeaway: A Gemini-based multi-agent system that generates, debates, and
  *evolves* hypotheses via ranking tournaments, producing proposals rated more
  novel by domain experts and validated in wet-lab follow-ups.
  Skill use: Supports a *tournament* prioritization design — generate many
  candidates, then rank them by pairwise debate/critique before emitting the top
  few, rather than scoring each in isolation.

## 8. Causal discovery / DAGs to surface untested relationships

- **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference*, 2nd
  edition. Cambridge University Press. ISBN 978-0-521-89560-6.
  Key takeaway: Structural causal models and DAGs make explicit which
  relationships are assumed, which are testable, and which are confounded; the
  do-calculus separates correlation from intervention-supported causation.
  Skill use: Grounds a "mechanism" generation move — represent the corpus's
  claimed effects as a DAG, then read off *untested* edges and confounds as
  candidate causal hypotheses (e.g., mediators of, or confounders behind, an
  observed accuracy difference).

## How this maps to the exploration skill

The references above imply a concrete four-step procedure.

1. **Ingest the hypothesis corpus.** Parse each recorded hypothesis with its
   status and verdict (confirmed / refuted / inconclusive / open), the *surviving
   rival explanations* it left standing, any *anomalies* (findings that
   contradicted the prevailing expectation), and any *exploratory observations*
   flagged as hypothesis-generating rather than confirmed (Tukey 1977;
   Wagenmakers et al. 2012). Treat the corpus as the Bayesian *prior* state
   (Lindley 1956; Rainforth et al. 2024).

2. **Apply generation moves.** Abduce candidate explanations for what the corpus
   shows (Peirce; Lipton 2004; Douven), using an explicit set of moves:
   *anomaly* (explain a finding that violates the paradigm — Kuhn 1962);
   *boundary/scope* (probe where a confirmed effect stops holding — Busse et al.
   2017); *mechanism* (propose/test a causal edge or confounder via a DAG —
   Pearl 2009); *contradiction* (reconcile two conflicting findings);
   *generalization/transfer* (map a confirmed relation to a new target by
   structural analogy — Gentner 1983); and *negation-of-refuted* (invert a
   refuted hypothesis into its testable complement). LLM-driven pipelines show
   these moves can be executed and self-critiqued for novelty/validity at scale
   (Yang et al. 2024; Lu et al. 2024; Gottweis et al. 2025).

3. **Prioritize candidates.** Rank by *expected information gain* about the open
   questions (Lindley 1956; Chaloner & Verdinelli 1995), using cheap EIG proxies
   where exact values are intractable (Foster et al. 2019), and fold in
   *testability*, *cost*, and *relevance* as a decision-theoretic utility
   (Chaloner & Verdinelli 1995). A tournament/debate ranking over the candidate
   slate (Gottweis et al. 2025) and the "loveliest, not merely likeliest —
   beware the best of a bad lot" check (Lipton 2004) select the final few.

4. **Emit drafts as new open hypotheses.** Output each survivor as a *new open*
   hypothesis record — statement, the deducible testable consequence, the
   generation move that produced it, and its priority rationale — for the
   separate, pre-registered testing skill to confirm (Wagenmakers et al. 2012;
   King et al. 2009). This skill never marks its own outputs confirmed; the
   generate→test→interpret loop is closed only by the confirmatory tooling in
   [methodology-references.md](methodology-references.md).
