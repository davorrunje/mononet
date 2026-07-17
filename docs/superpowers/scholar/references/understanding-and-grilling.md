# The Understanding Principle & the `grill` skill — grounding digest

**Date:** 2026-07-17 · **For:** `scholar` meta-spec §2.2 + the cross-cutting `grill` skill (sub-spec 1). · **Status:** verified-source digest; migrates to the plugin's `resources/references/`.

Companion to the [agency principle](agency-principle.md). Where agency says *the human decides and is accountable*, the **Understanding principle** says *the human must understand their own work and the methods they use, well enough to decide and defend* — and the plugin **grills and teaches** to ensure it. Goal: growth, not producing papers/theses the author cannot understand or defend.

## The principle

> Every material claim, decision, and method must be understood by the author to
> the standard a good mentor or reviewer would expect. The plugin verifies and
> **builds** that understanding through Socratic grilling and teaching; it never
> substitutes for it. A paper, thesis, or decision may not advance past something
> the author cannot explain or defend — silently.

## Why grilling works (and teaches)

| Effect | One-line | Source |
|---|---|---|
| **Illusion of explanatory depth** | People rate their understanding high until asked to explain step-by-step, then it collapses → self-assessed understanding is unreliable until probed. *The core justification.* | Rozenblit & Keil (2002), *Cognitive Science* 26(5):521–562, DOI:10.1207/s15516709cog2605_1 |
| **Testing effect / retrieval practice** | Retrieval (vs. re-reading) is itself learning → grilling *builds*, not just audits. | Roediger & Karpicke (2006), *Psych. Science* 17(3):249–255, DOI:10.1111/j.1467-9280.2006.01693.x |
| **Self-explanation effect** | Explaining to oneself yields principle-level, transferable understanding and self-detects gaps. | Chi et al. (1989), *Cognitive Science* 13(2):145–182, DOI:10.1207/s15516709cog1302_1 |
| **Desirable difficulties** | Harder-feeling retrieval/generation yields durable learning; fluency is a poor mastery proxy → calibrate grilling intensity, don't optimize for ease. | E. Bjork & R. Bjork (2011), *Psychology and the Real World* (book chapter) |
| **High-utility synthesis** | Practice testing + self-explanation rank high-utility across a large review. | Dunlosky et al. (2013), *PSPI* 14(1):4–58, DOI:10.1177/1529100612453266 |

Socratic questioning reference: Paul & Elder, *The Thinker's Guide to the Art of Socratic Questioning* (2007), ISBN 978-0944583319. *(The "Feynman technique" has no peer-reviewed primary source — folklore; lean on IOED + self-explanation.)*

## What a reviewer/examiner probes (the grilling agenda for "your work")

Original contribution / novelty vs. prior work; methodology justification ("why this method"); limitations & threats to validity (self-awareness is itself what's tested); real-time reasoning; "what would change your mind / falsify this?". Bluffing is penalized; "let me think" is rewarded. Sources: Petre & Rugg, *The Unwritten Rules of PhD Research*, 3e (2020), ISBN 9780335262120; viva guides (UCalgary "Nasty PhD Viva Questions"; Warwick Viva FAQs). Reviewer-rebuttal / self red-teaming is the pre-submission analog (practitioner guidance).

## Grilling the methodology (anti-cargo-cult)

Following rigor rituals without understanding defeats them. Grill the *why*, not just the *how*.

- **Conceptual vs. procedural knowledge** — procedures without conceptual anchoring are brittle/misapplied; the two develop iteratively. Hiebert & Lefevre (1986), reprint DOI:10.4324/9780203063538; Rittle-Johnson, Siegler & Alibali (2001), DOI:10.1037/0022-0663.93.2.346; Star (2005), *JRME* 36(5):404–411. Transfer requires understanding: NRC (2000), *How People Learn*, DOI:10.17226/9853.
- **"Cargo cult science"** — all the forms, none of the substance; "you must not fool yourself." Feynman (1974), Caltech commencement.
- **"Mindless statistics" / the null ritual** — mechanical NHST replacing statistical thinking. Gigerenzer (2004), *J. Socio-Economics* 33(5):587–606, DOI:10.1016/j.socec.2004.09.033.
- **ASA statement on p-values** — misuse from not understanding them. Wasserstein & Lazar (2016), *Am. Statistician* 70(2):129–133, DOI:10.1080/00031305.2016.1154108; "Beyond p<0.05" (2019), DOI:10.1080/00031305.2019.1583913.
- **Researcher degrees of freedom / garden of forking paths** — Simmons, Nelson & Simonsohn (2011), *Psych. Science* 22(11):1359–1366, DOI:10.1177/0956797611417632; Gelman & Loken (2013/2014). Motivation: Ioannidis (2005), *PLoS Med* 2(8):e124, DOI:10.1371/journal.pmed.0020124.

**Understanding checklist per rigor element** (grill seeds — can the author explain, not just perform?):
- *Preregistration / confirmatory-vs-exploratory:* fixes analysis before outcomes → curbs HARKing + shrinks forking paths; does NOT forbid exploration, only mislabeling it.
- *Rival hypotheses + discriminating tests:* name a competing explanation + a result that separates it.
- *Severity:* passing is evidence because the test would probably have failed if the claim were false (Mayo — cite explicitly if used).
- *Power / MDE:* set before data; a null from an underpowered study is uninformative, not evidence of no effect.
- *TOST / equivalence:* a non-significant p is NOT "no effect"; equivalence needs a pre-set bound. *(The most cargo-culted null claim.)*
- *Disclosure / file-drawer:* selective reporting is what makes p-values unreliable.
- *PRISMA:* a reconstructable audit trail vs. cherry-picking — not a diagram drawn after an ad-hoc search.
- *Explore/confirm firewall (`scholar`'s own):* it operationalizes forking-paths avoidance — letting exploration set the confirmatory test is the mechanism that invalidates it.

## Matt Pocock's "grill me" — same mechanic, inverted purpose

His skill ([marketplace](https://claudemarketplaces.com/skills/mattpocock/skills/grill-me); [writeup](https://www.aihero.dev/my-grill-me-skill-has-gone-viral)) interviews a developer to *elicit requirements they already hold*, one question at a time, and **lets the AI recommend answers** — the human is the ground truth. `scholar`'s grill verifies understanding of an **external** truth (claims/methods/literature); it **must not supply an answer key** for novel claims. Same machinery, opposite epistemics.

## Design (as settled in discussion)

**A cross-cutting `grill` skill** (companion to `progress`), a Socratic **tutor-examiner** whose goal is growth.

- **Targets:** `claim | cited-work | methodology`, with stage presets — `hypothesis-testing`→strategy (assumptions/entailments/falsifiers/rivals); `paper-synthesis`→positioning (novelty, citation support); thesis defensibility gate→full **mock viva**.
- **Loop:** probe → detect gap → **teach** (explain + link references) → re-probe (retrieval loop) → until it holds.
- **Two invocation paths:** (1) **self-invoked** on demand; (2) **automatic guardrail** at material-decision checkpoints (findings verdict, publish decision, thesis defensibility) and when a method is used whose rationale is undemonstrated.
- **Guardrail = stop, surface, offer to grill/teach, record — human may override, override logged.** Not a hard block (agency: the human drives; EU AI Act human-oversight allows override); but you cannot *silently* pass a gap. (Optionally the thesis defensibility gate could be made blocking — deferred open item.)

## Hard constraints (agency interlock)

- **Ask, don't grade substance.** Report "you could not explain X / had no answer for the falsification probe" (observed fact), never "you are wrong about X" (a judgment the AI isn't entitled to make on a novel claim).
- **Teaching is source-grounded, scoped by target.** *Methodology* + *cited-work* are largely established → teach freely from sources (these digests, the author's citations, external authoritative refs). *Novel claim* → teach how to reason/defend, cite, but never assert the claim's truth.
- **Settled vs. contested.** Grill hard on settled errors (non-significant p = "no effect"); on contested choices (frequentist vs. Bayesian) grill only for a *defensible justification* and present the standard critique — do not impose a school. Miscalibrating (treating contested as error) violates agency.
- **Calibrate depth to stakes** (severity/power logic): hardest on the method carrying the central claim. "Enough" = can state what it buys + its key assumption/limit + answer the one canonical critique. Anti-Goodhart: reward articulable understanding, not checklist completion — don't let the grill become its own ritual.

## Sources
Rozenblit & Keil 2002 (cog2605_1) · Roediger & Karpicke 2006 (01693.x) · Chi et al. 1989 (cog1302_1) · Bjork & Bjork 2011 · Dunlosky et al. 2013 (1529100612453266) · Paul & Elder 2007 (ISBN 978-0944583319) · Petre & Rugg 2020 (ISBN 9780335262120) · Pocock grill-me (marketplace + aihero.dev) · Hiebert & Lefevre 1986 (10.4324/9780203063538) · Rittle-Johnson et al. 2001 (0022-0663.93.2.346) · Star 2005 (JRME 36(5)) · NRC 2000 (10.17226/9853) · Feynman 1974 (calteches.library.caltech.edu/51/2/CargoCult.htm) · Gigerenzer 2004 (socec.2004.09.033) · Wasserstein & Lazar 2016 (00031305.2016.1154108) · Wasserstein et al. 2019 (00031305.2019.1583913) · Simmons et al. 2011 (0956797611417632) · Gelman & Loken 2013/2014 · Ioannidis 2005 (pmed.0020124).

## Flags
Feynman technique = folklore (no primary). Pressley et al. 1987 elaborative-interrogation DOI unverified (confirmed via Dunlosky 2013). Star 2005 + Gelman & Loken 2014 DOIs unverified (stable URLs used). Severity (Mayo) not in the source set — cite Mayo explicitly if used. Reviewer red-teaming = practitioner guidance, not experimental.
