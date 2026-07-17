# Positioning / Related-Works Synthesis — methodology digest

**Date:** 2026-07-17 · **For:** `scholar` `literature` capability, `position` mode (sub-spec 2). · **Status:** verified-source digest; migrates to the plugin's `resources/references/`.

For a committed paper: systematically establish novelty, select baselines, situate the contribution. Every named method has a verified identifier below.

## 1. Named methodologies

- **Systematic Literature Review (SLR)** — pre-registered, auditable: review questions → search → inclusion/exclusion → screen → extract → synthesize. Kitchenham & Charters (2007), *Guidelines for Performing SLRs in Software Engineering*, EBSE-2007-01. The backbone the others plug into (plan/conduct/report).
- **PRISMA** (reporting standard, not a search method) — 27-item checklist + **flow diagram** (identified → deduped → screened → excluded-with-reasons → included). Page et al. (2021), *BMJ* 372:n71, DOI:10.1136/bmj.n71; Moher et al. (2009), *PLoS Med* 6(7):e1000097, DOI:10.1371/journal.pmed.1000097. Durable value = the flow diagram's **audit trail** (the anti-cherry-picking defense).
- **Snowballing** — Wohlin (2014), EASE, DOI:10.1145/2601248.2601268. Diverse **start set** (Google Scholar to reduce publisher bias); **backward** (references → foundations/precedent) + **forward** (citations → newer competitors/SOTA); iterate to **saturation**.
- **Concept-centric matrix** — Webster & Watson (2002), *MIS Quarterly* 26(2):xiii–xxiii. Do **not** organize author-by-author; organize **concept × source** (rows=papers, cols=attributes your delta turns on). This matrix *is* the comparison table you ship.
- **Review typology** — Cooper (1988), *Knowledge in Society* 1(1):104–126, DOI:10.1007/BF03177550 `[page range unverified]`. Choose the organizing spine; for a methods paper a **conceptual/thematic** taxonomy beats chronological.
- **Vote-counting / feature matrix** — weak for evidence claims, legitimate for a **capability comparison table** (method × feature, ✓/✗/partial); the artifact reviewers scan first.
- **Related-Work section structure** — thematic grouping, one paragraph per approach-family, each with an explicit **delta**: "Unlike ⟨family⟩ we ⟨difference⟩, which yields ⟨consequence⟩." Contribution as a bulleted delta in intro, mirrored in related work.

## 2. Process: committed topic → `positioning.md`

1. **Frame the claim(s)** as 1–3 falsifiable delta statements *before* searching (the review questions).
2. **Seed set** — 3–6 diverse anchors spanning communities/terminologies.
3. **Snowball to saturation** — backward + forward (S2/OpenAlex for forward); log every candidate with include/exclude reason (PRISMA counts). Stop when no new relevant methods appear.
4. **Screen** against explicit inclusion criteria; record exclusions with reasons.
5. **Extract into the concept matrix** — rows = included methods; cols = the concepts your delta turns on.
6. **Derive the taxonomy** from matrix column clusters (the section spine).
7. **Write the delta** — per taxonomy branch, one sentence grounded in a matrix cell.
8. **Derive baselines from the taxonomy**, not from convenience.
9. **Ship `positioning.md`**: taxonomy, comparison table, PRISMA search log, novelty/delta bullets, baseline list with justification.

**Baseline selection:** one strong tuned representative **per taxonomy branch**; current **SOTA** on target datasets; the **most-likely-cited-against-you**; a **simplest reasonable** floor. Match conditions (splits, tuning budget, compute). Under-tuned baselines are the #1 reviewer complaint (Dacrema et al. 2019; Musgrave et al. 2020; Lin 2019).

**Defend against "already done":** adversarial precedent search per contribution bullet (forward-snowball nearest prior work + keyword on the *mechanism*); maintain a **"closest prior work"** paragraph with the precise testable delta; encode the delta as an **ablation**; keep the PRISMA log.

## 3. Anti-patterns → safeguards

| Anti-pattern | Safeguard |
|---|---|
| Cherry-picking related work | PRISMA log w/ exclusion reasons; diverse start set |
| Author-by-author annotated bibliography | concept-centric matrix (Webster & Watson) |
| Overclaiming novelty | adversarial precedent search; "closest prior work" + delta; isolating ablation |
| Weak/untuned baselines | equal tuning budget, matched splits; SOTA + strong per branch |
| Missing a key baseline | baselines derived from the taxonomy, not convenience |
| Single-community myopia | forward-snowball across venues; seed set spans terminologies |
| Non-reproducible comparison | report HPs, seeds, splits, search date/version |

**Venue expectation (NeurIPS/ICML):** reviewers judge novelty/significance + relation to prior work; checklists require limitations + reproducibility. The positioning doc should pre-answer "delta over closest work?" and "baselines strong and fairly tuned?" — the two rejection levers.

## 4. Level split — same method, two configurations

| Param | Hypothesis level ("would a reviewer say this is known?") | Paper level (full related work) |
|---|---|---|
| Scope | one claim; narrow | all contributions; whole design space |
| Stopping | claim shown answered/unanswered | snowball saturation across branches |
| Direction | adversarial backward+forward from 1–3 nearest | balanced coverage; full taxonomy |
| Output | verdict + the papers that would reject it + surviving delta | taxonomy + table + prose + baselines |

→ one skill, `mode/depth` switch; hypothesis = a fast adversarial precedent *rapid review*. Ship a PRISMA-style log in both modes.

## 5. Tooling (free/scriptable emphasized)
- Forward snowball + metadata: **Semantic Scholar API**, **OpenAlex**, **Crossref** (DOIs). Google Scholar = best unbiased start set, no API.
- Screening/active learning: **ASReview LAB** (van de Schoot et al. 2021, *Nat. Mach. Intell.* 3:125–133, DOI:10.1038/s42256-020-00287-7); **Rayyan**.
- Reference mgmt/dedup: **Zotero** (+ Better BibTeX); **Publish or Perish**.
- PRISMA artifacts: **PRISMA2020** R package / official flow-diagram generator.

## Sources
Kitchenham & Charters 2007 (EBSE-2007-01) · Page et al. 2021 (bmj.n71) · Moher et al. 2009 (pmed.1000097) · Wohlin 2014 (2601248.2601268) · Webster & Watson 2002 (MISQ 26(2)) · Cooper 1988 (BF03177550) · van de Schoot et al. 2021 (s42256-020-00287-7) · Dacrema et al. 2019 (RecSys, 3298689.3347058) · Musgrave et al. 2020 (arXiv:2003.08505) · Lin 2019 (SIGIR Forum, 3308774.3308781).

## Caveats
Cooper 1988 page range not verified in primary this session. NeurIPS/ICML checklist wording paraphrased (changes yearly) — verify current-year text before quoting.
