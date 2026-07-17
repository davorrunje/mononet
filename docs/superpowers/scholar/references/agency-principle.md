# Agency Principle — "Assistants, Not Researchers" (grounding digest)

**Date:** 2026-07-17
**For:** `scholar` plugin (meta-spec §2.1) — the single most important guiding principle.
**Status:** Verified-source digest. Migrates to the plugin's `resources/references/`.

> This is a curated, primary-source digest. It exists so the agency principle is
> recorded as a *grounded* rule, not an opinion. All identifiers were verified at
> compile time except where flagged.

## The principle

> The `scholar` skills are **assistants, not researchers.** They keep the
> accounts of a research program, advise as a mentor, and discuss as a colleague,
> but they do **not** perform independent research and do **not** make material
> scientific decisions. Every material decision — is a hypothesis confirmed /
> refuted; is a result real; is a paper worth publishing; what the thesis claims;
> is it defensible — is the human researcher's, **recorded with a named human
> sign-off**. You author; the skill drafts. Automation covers the mechanical and
> mnemonic, never the judgemental. The researcher is in the driving seat.

It is **not** an anti-AI stance. Every norm below *permits* AI assistance
(editing, retrieval, bookkeeping, discussion) while forbidding AI *authorship*
and AI *decision-making*. The principle draws that bright line and keeps the
skills useful on the permitted side while structurally refusing to cross it.

## Why it holds — grounded

### 1. Authorship & accountability: accountability is non-delegable and attaches only to humans

- **ICMJE** — authorship requires final approval *and accountability*; AI tools
  cannot be authors because they cannot take responsibility, declare conflicts,
  or hold copyright; humans remain responsible for AI-produced content.
- **COPE position statement, "Authorship and AI tools" (13 Feb 2023)** — AI
  cannot be listed as an author; use must be disclosed; authors are fully
  responsible.
- **Nature/Springer Nature** — LLMs fail authorship criteria because
  "authorship carries accountability… which cannot be effectively applied to
  LLMs."
- **Science/AAAS** — Thorp, "ChatGPT is fun, but not an author," *Science*
  379(6630):313 (2023); undisclosed AI-generated text treated as plagiarism.
- **Elsevier / Wiley / T&F / SAGE** — same consensus: only humans/legal persons
  can hold accountability.
- **ML venues** — **ICML 2023**: text produced entirely by LLMs prohibited
  (except as experimental object); polishing author text allowed; the human is
  "ultimately responsible." **NeurIPS**: LLMs allowed as tools but must be
  described; responsibility stays with authors. **ACL 2023**: language/search
  assistance OK, but authors must read/verify every reference; errors and
  plagiarism are the author's responsibility.
- **CRediT (ANSI/NISO Z39.104-2022)** — 14 contributor roles attribute
  contribution to *named people*, not tools.

### 2. Human oversight / human-in-command: decision authority stays human

- **EU AI Act, Article 14 (human oversight)** — high-risk AI must be designed
  for oversight by natural persons who can interpret output, override it, or
  *decide not to use it*, and must "remain aware of… over-relying on the output
  (automation bias)." Ties oversight directly to automation bias.
- **Santoni de Sio & van den Hoven (2018), "Meaningful Human Control over
  Autonomous Systems," *Frontiers in Robotics and AI* 5:15** — grounds control
  in *tracking* (responsive to human reasons) and *tracing* (a human who
  understands and is answerable). "Recorded with a human sign-off" is the
  tracing condition made concrete.

### 3. Automation bias: why the skill must halt and ask, not decide

- **Parasuraman & Riley (1997), "Humans and Automation: Use, Misuse, Disuse,
  Abuse," *Human Factors* 39(2):230–253** — canonical taxonomy; "misuse" =
  over-reliance causing monitoring failure and decision bias.
- **Skitka, Mosier & Burdick (1999), *Int. J. Human-Computer Studies*
  51(5):991–1006** — empirical automation bias: people accept incorrect
  automated advice over available contradictory evidence.
- **Research-specific failure mode** — trusting an LLM to *decide* (vs assist)
  injects fabricated citations; audits report high fabrication rates in
  LLM-generated references. *(Some figures come from 2026 preprints — treat as
  illustrative; verify final-venue status before formal use.)*

### 4. Research-integrity frameworks place accountability on the researcher

- **Singapore Statement on Research Integrity (2010)** — four principles:
  honesty, **accountability**, professionalism, stewardship.
- **ALLEA, European Code of Conduct for Research Integrity (Revised 2023)** —
  the EU's primary integrity standard; the 2023 revision addresses digital/AI
  tools and places responsibility for good practice on researchers.

### 5. "Augment, not replace" — the positive framing

- **Engelbart (1962), "Augmenting Human Intellect: A Conceptual Framework"
  (SRI AFOSR-3223)** — tools amplify human capability within a human-directed
  system.
- **Bush (1945), "As We May Think," *The Atlantic*** — the memex aids memory and
  association, not judgement.

## Guardrails that follow (design-enforceable)

1. **Named human sign-off on every material decision** (hypothesis verdict,
   result-is-real, publishable, thesis claim, defensibility). Record the human,
   not the skill. *(Singapore/ALLEA accountability + ICMJE final-approval +
   meaningful-human-control "tracing".)*
2. **Skill halts at judgement points and asks.** The human must be able to
   disregard the tool. *(EU AI Act Art. 14; automation-bias literature.)*
3. **Drafts are proposals, never authored text.** The researcher reads,
   verifies, and adopts. *(ICMJE/COPE/Science/Nature.)*
4. **No unattended end-to-end paper/thesis generation.** *(Science = plagiarism;
   ICML = prohibited.)*
5. **Verify every AI-surfaced reference before use.** *(ACL 2023; fabrication
   rates.)*
6. **Disclose and log AI use** (the mechanical/mnemonic assistance) — feeds the
   audit trail the skills already maintain. *(COPE/Nature/ICML/NeurIPS.)*
7. **Automate only the mechanical and mnemonic** — bookkeeping, formatting,
   retrieval, roll-ups, reminders; never the judgemental.

## The bright line (nuance to preserve)

Every policy cited permits AI *assistance* and forbids AI *authorship /
decision-making*. The principle is that line, not a ban. One caveat: the
"editing/polishing without disclosure" carve-out (Nature) sits close to the
line — `scholar` should **default to disclosure** to stay clearly on the safe
side.

## Sources

- ICMJE — Role of Authors and Contributors: <https://www.icmje.org/recommendations/browse/roles-and-responsibilities/defining-the-role-of-authors-and-contributors.html>; AI use by authors: <https://www.icmje.org/recommendations/browse/artificial-intelligence/ai-use-by-authors.html>
- COPE — Authorship and AI tools (2023-02-13): <https://publicationethics.org/guidance/cope-position/authorship-and-ai-tools>
- Nature Portfolio — AI policy: <https://www.nature.com/nature-portfolio/editorial-policies/ai>
- Thorp, H.H. (2023), *Science* 379(6630):313, DOI 10.1126/science.adg7879
- Elsevier publishing-ethics: <https://www.elsevier.com/about/policies-and-standards/publishing-ethics> *(not directly fetched; corroborated via secondary sources)*
- ICML 2023 LLM policy: <https://icml.cc/Conferences/2023/llm-policy>
- NeurIPS LLM policy: <https://neurips.cc/Conferences/2025/LLM>
- ACL 2023 Policy on AI Writing Assistance: <https://2023.aclweb.org/blog/ACL-2023-policy/>
- CRediT (ANSI/NISO Z39.104-2022): <https://credit.niso.org> · <https://www.niso.org/publications/z39104-2022-credit>
- EU AI Act Art. 14 (Human oversight): <https://artificialintelligenceact.eu/article/14/>
- Santoni de Sio & van den Hoven (2018), *Front. Robot. AI* 5:15, DOI 10.3389/frobt.2018.00015
- Parasuraman & Riley (1997), *Human Factors* 39(2):230–253, DOI 10.1518/001872097778543886
- Skitka, Mosier & Burdick (1999), *IJHCS* 51(5):991–1006, DOI 10.1006/ijhc.1999.0252
- Singapore Statement on Research Integrity (2010): <https://wcrif.org/guidance/singapore-statement>
- ALLEA European Code of Conduct for Research Integrity (2023): <https://allea.org/code-of-conduct/>
- Engelbart (1962), Augmenting Human Intellect: <https://www.dougengelbart.org/pubs/augment-3906.html>
- Bush (1945), As We May Think, *The Atlantic* *(canonical; not re-fetched)*

## Verification notes

Verified this session: ICMJE, COPE, Nature, Science (DOI), CRediT/NISO, EU AI
Act Art. 14, ICML/NeurIPS/ACL, Singapore Statement, ALLEA 2023, Parasuraman &
Riley, Skitka et al., Santoni de Sio & van den Hoven, Engelbart. Not directly
fetched: Elsevier policy page (corroborated), Bush 1945 (canonical). LLM
citation-fabrication figures come partly from 2026 preprints — illustrative;
confirm final-venue status before formal citation.
