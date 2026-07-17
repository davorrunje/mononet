# Mentor / Reviewer Personas — grounding digest

**Date:** 2026-07-17 · **For:** `scholar` `grill` skill + the "advise as a mentor" voice (meta-spec §3.7, sub-spec 1). · **Status:** verified-source digest; migrates to the plugin's `resources/references/`.

**Bottom line:** style *typologies* are well-described; matching a mentor to the author's *personality* is **not** evidence-based (the learning-styles myth); and an AI *inferring* personality to pick a voice is unreliable and violates agency. The defensible design is **author-selectable personas + situational/task calibration — never system-inferred personality.**

> One-line rule for the spec: **match the voice to the task and the author's stated choice — never to an inferred personality.**

## 1. Style typologies (descriptive — solid)

- **Gatfield (2005)** — structure × support 2×2: *laissez-faire / pastoral / directorial / contractual*. *J. Higher Education Policy & Management* 27(3):311–325, DOI:10.1080/13600800500283585.
- **Lee (2008)** — five concepts of supervision: *functional, enculturation, critical thinking, emancipation, relationship development*. Best source for a reviewer-voice spectrum (critical-thinking/emancipation ≈ the grill examiner). *Studies in Higher Education* 33(3):267–281, DOI:10.1080/03075070802049202.
- **Kram (1985)** — mentoring functions: *career* (challenge, sponsorship) vs *psychosocial* (role-modeling, acceptance). ISBN 978-0673156174. **Higgins & Kram (2001)** — developmental *networks* (a portfolio, not one dyad), *AMR* 26(2):264–288, DOI:10.5465/amr.2001.4378023 *(verify)*.
- **Gurr (2001)** — "rackety bridge": dynamically align directiveness (hands-on↔hands-off) to the student's growing *competent autonomy*. The developmental-matching model, doctoral-specific. *HERD* 20(1):81–92, DOI:10.1080/07924360120043882.

## 2. Feedback science (efficacy — solid)

- **Hattie & Timperley (2007)**, "The Power of Feedback," *Review of Educational Research* 77(1):81–112, DOI:10.3102/003465430298487. Effective feedback answers *where am I going / how am I going / where next*; feedback at the **self/praise level is ineffective or harmful**, at the **process/self-regulation level most effective**. → the real axis is **challenge the argument/process, not the person** — not "harsh vs. nice."

## 3. Matching to the author — grounded vs. myth (the crux)

- **Learning-styles matching is a MYTH.** Pashler, McDaniel, Rohrer & Bjork (2008/09), *PSPI* 9(3):105–119, DOI:10.1111/j.1539-6053.2009.01038.x — the "meshing hypothesis" fails the crossover-interaction test that would validate it. Nancekivell et al. (2020), *JEP* 112(2):221–235, DOI:10.1037/edu0000366 (myth is pervasive + essentialized). **Do not build trait-based persona auto-assignment.**
- **Big Five ↔ style "fit": weak/mixed** — mostly main effects (agreeable/conscientious mentors → better relationships), no reliable *matching* interactions. Not actionable matching science.
- **Better-grounded alternatives:**
  - **Situational/developmental** — calibrate directiveness to competence+stage (Gurr 2001; Hersey & Blanchard SLT, ISBN 978-0132556408 — *SLT's own evidence is mixed; use as scaffold, not mechanism*).
  - **Self-Determination Theory autonomy-support** — the robust constant posture: support autonomy, competence, relatedness. Ryan & Deci (2000), *American Psychologist* 55(1):68–78, DOI:10.1037/0003-066X.55.1.68.
  - **Desirable difficulties / calibrated challenge** — Bjork & Bjork (2011). Keeps the grill's productive-struggle bite.
- **Supervisor–student fit matters — but it's relationship/expectation alignment, not personality matching** — mismatch drives attrition/dissatisfaction (Sverdlik et al., *IJDS* 13:361–388).

## 4. Ethics — no AI personality inference

LLM personality inference is contested on validity and ethically fraught (consent, bias, manipulation): arXiv:2503.02082, arXiv:2405.13052. A system that silently profiles the author to pick a voice is both unreliable and an agency/consent violation. **The author chooses; the system does not psychoanalyze.**

## 5. Design (as settled)

**A small set of author-selectable personas** (derived from Lee × Gatfield, not personality theory):

1. **Sounding board** — pastoral / relationship-development / high autonomy-support; early, exploratory.
2. **Critical examiner** — the **default grill** (critical-thinking + emancipation; calibrated difficulty).
3. **Directive editor** — directorial / functional; concrete, process-level (Hattie & Timperley), deadline-driven.
4. *(opt-in)* **Devil's advocate** — time-boxed; explicitly challenge-to-the-argument.

**Chosen by three author-controllable levers, never inferred:**
1. **Self-selected** (default) — the author picks. Satisfies agency; sidesteps the myth + inference ethics.
2. **Task/stage-suggested** — a *suggested* default keyed to the visible artifact/stage (early draft → sounding board; near-submission → examiner), à la Gurr; always overridable; keyed to the *work*, not the *person*.
3. **Feedback-calibrated** — adjusts to explicit author feedback ("too harsh" / "push harder"); feedback content always targets process/argument, not self.

**Invariants:** autonomy-support (SDT) is constant across all personas; only *directiveness* and *challenge-intensity* vary. Hidden personality profiling is **explicitly forbidden**. The grill keeps its bite (Bjork), but the author holds the difficulty dial (agency + Understanding principle).

## Sources
Gatfield 2005 (13600800500283585) · Lee 2008 (03075070802049202) · Kram 1985 (ISBN 978-0673156174) · Higgins & Kram 2001 (amr.2001.4378023, verify) · Gurr 2001 (07924360120043882) · Hattie & Timperley 2007 (003465430298487) · Pashler et al. 2008/09 (j.1539-6053.2009.01038.x) · Nancekivell et al. 2020 (edu0000366) · Ryan & Deci 2000 (0003-066X.55.1.68) · Hersey & Blanchard (ISBN 978-0132556408) · Bjork & Bjork 2011 · Sverdlik et al. (IJDS 13:361–388) · AI personality inference (arXiv:2503.02082, arXiv:2405.13052).

## Flags
Higgins & Kram DOI worth confirming. Pashler et al. 2008/2009 date ambiguity (one paper). Hersey–Blanchard SLT and Big-Five-matching are where popular claims outrun evidence — treated as such. Big-Five/fit sources are representative secondary literature, not single authoritative primaries.
