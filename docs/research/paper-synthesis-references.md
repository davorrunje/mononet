# Paper-synthesis references

This document collects verified primary sources for a *paper-synthesis* research
skill: one that maintains a **claim→evidence ledger** for an academic paper.
Each paper claim is bound to the hypotheses and experiments that back it and to
committed result tables; each binding carries a status (supported / thin / gap /
contradicted) and a narrative role (headline / null-result / limitation). The
skill emits a **gap list** of under-supported claims and keeps every
quantitative claim in sync with committed results (result tables injected from a
build step). It is the third of a trio: the confirmatory half lives in
[methodology-references.md](methodology-references.md) (pre-registration, strong
inference, severity, `rliable`, equivalence, reproducibility checklists) and the
generative half in
[hypothesis-generation-references.md](hypothesis-generation-references.md)
(abduction, EDA, Bayesian optimal experimental design, automated discovery).
This file does **not** repeat those; it cross-links them where the argument
touches confirmation or generation. Because the output feeds a methodology
section, citations must be real and verified — every entry below was checked
against its publisher of record, arXiv, or a stable archive; none is marked
UNVERIFIED.

## 1. Argument structure for claims — the Toulmin model

- **Toulmin, S. E. (1958; updated ed. 2003).** *The Uses of Argument.*
  Cambridge University Press. Updated edition ISBN 978-0-521-53483-3;
  [Cambridge Core](https://www.cambridge.org/core/books/uses-of-argument/26CF801BC12004587B66778297D5567C).
  Key takeaway: A substantial argument is not a bare premise→conclusion step but
  a six-part layout — **claim** (C), **grounds/data** (D), **warrant** (W)
  licensing the inference, **backing** (B) supporting the warrant, a **qualifier**
  (Q) stating the force ("presumably", "necessarily"), and a **rebuttal** (R)
  giving conditions of exception — and the standards of soundness are
  field-dependent.
  Skill use: The Toulmin sextet is the schema for each ledger row — the paper
  *claim* is C, the cited experiments/result tables are D, the analysis method
  that turns data into the claim is W, the methodology references are B, the
  status/role flags encode Q, and the limitation is R. "Thin"/"gap" statuses are
  precisely a missing or weak W/B.

## 2. Fine-grained claim↔evidence linking in scholarly communication

These two models are the closest existing formalizations of a claim→evidence
ledger: both represent scientific assertions and their supporting evidence as
first-class, machine-addressable objects rather than free text.

- **Clark, T., Ciccarese, P. N., & Goble, C. A. (2014).** *Micropublications: a
  semantic model for claims, evidence, arguments and annotations in biomedical
  communications.* Journal of Biomedical Semantics 5: 28.
  DOI: [10.1186/2041-1480-5-28](https://doi.org/10.1186/2041-1480-5-28);
  arXiv: [1305.3506](https://arxiv.org/abs/1305.3506).
  Key takeaway: A micropublication is a formal argument graph linking a claim to
  its supporting evidence, methods, and to challenging or qualifying statements,
  so that the *warrant* for a claim (not just a citation) is machine-traceable
  and the whole support structure can be audited.
  Skill use: Direct prior art for the ledger — each row is a micropublication in
  miniature (claim + evidence + method + rebuttal), and the "contradicted" status
  is exactly a modeled challenging statement.

- **Groth, P., Gibson, A., & Velterop, J. (2010).** *The anatomy of a
  nanopublication.* Information Services and Use 30(1–2): 51–56.
  DOI: [10.3233/ISU-2010-0613](https://doi.org/10.3233/ISU-2010-0613).
  Key takeaway: A nanopublication packages a single assertion together with its
  **provenance** (how/where it was derived) and **publication info** as a set of
  named RDF graphs, making the atomic scientific statement independently
  citable, attributable, and aggregatable.
  Skill use: Motivates giving every ledger claim a stable identifier plus a
  provenance pointer to the exact result table / run that supports it — the
  substrate for the consistency check in §7 below.

## 3. Reproducible research — linking claims to computations

The consistency check (§7) depends on a claim being tied to the *computation*
that produced its number; this literature is the foundation for that binding.
See also the ML-specific reproducibility standards (Pineau et al. 2021; Gebru et
al. 2021) in [methodology-references.md](methodology-references.md) §6.

- **Knuth, D. E. (1984).** *Literate Programming.* The Computer Journal 27(2):
  97–111. DOI: [10.1093/comjnl/27.2.97](https://doi.org/10.1093/comjnl/27.2.97).
  Key takeaway: Interleave the explanatory prose and the executable code in one
  source, ordered for human comprehension, so that the narrative and the
  computation are a single artifact that cannot silently drift apart.
  Skill use: The intellectual root of injecting result tables from a build step —
  prose and computation share one source of truth rather than being copied by
  hand.

- **Gentleman, R., & Temple Lang, D. (2007).** *Statistical Analyses and
  Reproducible Research.* Journal of Computational and Graphical Statistics
  16(1): 1–23.
  DOI: [10.1198/106186007X178663](https://doi.org/10.1198/106186007X178663).
  Key takeaway: A "compendium" of dynamic documents binds text, code, and data
  so that every reported number can be regenerated and verified from the same
  source, and authors can reproduce their own results later.
  Skill use: Formalizes the build-step contract — the paper's numbers are
  *derived*, not transcribed, which is what makes a staleness check well-defined.

- **Peng, R. D. (2011).** *Reproducible Research in Computational Science.*
  Science 334(6060): 1226–1227.
  DOI: [10.1126/science.1213847](https://doi.org/10.1126/science.1213847).
  Key takeaway: Reproducibility — the ability to recompute published results
  from the authors' code and data — is a minimum standard for judging
  computational claims when full independent replication is infeasible.
  Skill use: Sets the bar the ledger enforces at claim level: a quantitative
  claim is only "supported" if its number is recomputable from committed
  code + data.

- **Marwick, B., Boettiger, C., & Mullen, L. (2018).** *Packaging Data
  Analytical Work Reproducibly Using R (and Friends).* The American Statistician
  72(1): 80–88.
  DOI: [10.1080/00031305.2017.1375986](https://doi.org/10.1080/00031305.2017.1375986).
  Key takeaway: A **research compendium** organizes data, analysis code,
  computational environment, and manuscript in one version-controlled container
  with clear separation of inputs, methods, and outputs.
  Skill use: The compendium layout is the file-system model the skill assumes —
  committed result tables as the shared output layer that both the paper prose
  and the ledger read from.

## 4. Continuously-built / living manuscripts

- **Himmelstein, D. S., Rubinetti, V., Slochower, D. R., Hu, D., Malladi, V. S.,
  Greene, C. S., & Gitter, A. (2019).** *Open collaborative writing with
  Manubot.* PLOS Computational Biology 15(6): e1007128.
  DOI: [10.1371/journal.pcbi.1007128](https://doi.org/10.1371/journal.pcbi.1007128);
  project: <https://manubot.org/>.
  Key takeaway: Manubot builds a manuscript from plain-text source under version
  control via a continuous-integration pipeline, pulling citations and content
  programmatically so the rendered paper is a reproducible build artifact rather
  than a hand-maintained document.
  Skill use: The reference implementation of "results flow from code into the
  paper" — the CI build that injects committed result tables is exactly the step
  the consistency check (§7) guards; a claim goes **stale** when the built
  numbers no longer match the prose.

## 5. Scientific writing craft (optional prose-drafting stage)

- **Gopen, G. D., & Swan, J. A. (1990).** *The Science of Scientific Writing.*
  American Scientist 78(6): 550–558.
  Stable URL: <https://www.jstor.org/stable/29774235>.
  Key takeaway: Reader comprehension is governed by structural expectations —
  place the topic/old information at the start of a sentence and the new,
  emphasis-bearing information in the "stress position" at the end; align the
  unit of discourse with the grammatical subject.
  Skill use: Rules the optional prose-drafting stage can apply mechanically when
  turning a supported ledger row into a sentence — claim as subject, evidence in
  the stress position.

- **Schimel, J. (2012).** *Writing Science: How to Write Papers That Get Cited
  and Proposals That Get Funded.* Oxford University Press.
  ISBN 978-0-19-976024-4;
  [OUP](https://global.oup.com/academic/product/writing-science-9780199760244).
  Key takeaway: Effective scientific writing is structured as a story
  (opening–challenge–resolution; the OCAR/ABDCE arcs), with the paper's central
  claim framed as the resolution the evidence delivers.
  Skill use: Supplies the narrative-role vocabulary — headline claims are the
  story's resolution, null results and limitations are its complications — which
  the skill's role flags (headline / null-result / limitation) encode.

## 6. Research gaps / framing contributions (the gaps verb)

- **Sandberg, J., & Alvesson, M. (2011).** *Ways of constructing research
  questions: gap-spotting or problematization?* Organization 18(1): 23–44.
  DOI: [10.1177/1350508410372151](https://doi.org/10.1177/1350508410372151).
  Key takeaway: Most research questions are built by *gap-spotting* (confusion,
  neglect, or application spotting); the authors argue this under-produces
  influential work and advocate *problematization* — challenging the assumptions
  underlying existing literature — as the higher-value alternative.
  Skill use: Disciplines the **gaps verb**: distinguish a genuine evidential gap
  (a claim lacking backing, to be reported honestly) from mere rhetorical
  gap-spotting, and flag whether an emitted gap is a support deficiency or an
  assumption worth problematizing. Complements the generative gap-finding in
  [hypothesis-generation-references.md](hypothesis-generation-references.md).

## 7. Reporting and valuing null/negative results

Kept brief; the statistical machinery for *warranting* a null (equivalence
testing / TOST, power, minimum detectable effect, severity) lives in
[methodology-references.md](methodology-references.md) §§3, 5 (Mayo 2018;
Lakens 2017/2018; Altman & Bland 1995) and is not repeated here.

- **Rosenthal, R. (1979).** *The "File Drawer Problem" and Tolerance for Null
  Results.* Psychological Bulletin 86(3): 638–641.
  DOI: [10.1037/0033-2909.86.3.638](https://doi.org/10.1037/0033-2909.86.3.638).
  Key takeaway: Selective non-publication of null results ("the file drawer")
  biases the literature toward false positives; a robust body of evidence must
  count and report the nulls, not bury them.
  Skill use: Justifies the ledger's first-class **null-result** narrative role —
  a well-supported null is a reportable contribution, not an absence — and, with
  the equivalence entries in the companion file, sets the higher evidential bar a
  headline null claim must clear before it is marked "supported."

## How this maps to the skill

**Claim→evidence ledger schema.** Each row is a Toulmin sextet (Toulmin 1958):
claim / grounds (committed result tables) / warrant (analysis method) / backing
(methodology refs) / qualifier (status flag) / rebuttal (limitation). The
micropublication and nanopublication models (Clark et al. 2014; Groth et al.
2010) are the direct prior art for representing that structure as addressable,
provenance-carrying objects, and give every claim a stable identifier plus a
pointer to the exact run/table that backs it.

**Gaps verb.** Sandberg & Alvesson (2011) separate genuine evidential gaps from
rhetorical gap-spotting, so the emitted gap list distinguishes "claim lacks
backing" from "assumption worth problematizing." The generative counterpart —
turning a gap into the next hypothesis to test — is deferred to
[hypothesis-generation-references.md](hypothesis-generation-references.md).

**Consistency / staleness check.** The claim is bound to the computation that
produced its number (Knuth 1984; Gentleman & Temple Lang 2007; Peng 2011;
Marwick et al. 2018), and the paper is a continuously built artifact whose result
tables are injected from a build step (Himmelstein et al. 2019). A quantitative
claim is **stale** (or **contradicted**) when the freshly built numbers diverge
from the prose; it is "supported" only when recomputable from committed
code + data.

**Optional prose drafting.** When a row is well-supported, Gopen & Swan (1990)
and Schimel (2012) govern rendering it into prose — claim in the subject
position, evidence in the stress position, narrative role (headline /
null-result / limitation) mapped onto the story arc. A well-supported null is
reported as a contribution, not hidden (Rosenthal 1979), with its statistical
warrant drawn from the equivalence/severity entries in the companion file.
