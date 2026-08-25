# Data request — De Clercq batch-settling concentration profiles

Draft email to request the numeric `C(z,t)` field behind Figs 5.8/5.10 of the
De Clercq (2006) PhD thesis. Recipients (verified addresses):

- **Prof. Jeriffa De Clercq** — `Jeriffa.DeClercq@UGent.be` — **first author**; ran
  the experiments and produced the profiles, so the primary contact. Now Associate
  Professor, Faculty of Engineering & Architecture, Dept. of Materials, Textiles
  and Chemical Engineering (EA11), Ghent University.
- **Ingmar Nopens** — `Ingmar.Nopens@UGent.be` (BIOMATH, Ghent) — co-author, active.
- **Peter A. Vanrolleghem** — `peter.vanrolleghem@gci.ulaval.ca` (modelEAU, Laval) — co-author, active.

Send **To:** De Clercq, **Cc:** Nopens + Vanrolleghem.

---

**Subject:** Request: numeric batch-settling solids-concentration profiles (De Clercq 2005 / 2006 thesis, Figs 5.8 & 5.10)

Dear Prof. De Clercq, (cc Prof. Nopens and Prof. Vanrolleghem),

I am working on structure-preserving neural networks for scalar conservation laws —
specifically networks that are *monotone by construction*, so that a reconstructed
solution field automatically satisfies the physical admissibility (monotone
profile) of the underlying PDE. Batch sedimentation (Kynch theory) is an ideal
real-world test case: the solids concentration profile is monotone in height by
physics, governed by a scalar conservation law.

Your radiotracer measurements — the dense `C(z,t)` fields in **Figures 5.8 and 5.10
of your 2006 PhD thesis** (*Batch and continuous settling of activated sludge:
in-depth monitoring and 1D compression modelling*), also in De Clercq et al.,
*Water Research* 39(10), 2005 — are exactly what I need. The thesis presents them
as 3-D surface plots, so I cannot recover the numbers from the document.

Would you be willing to share the **numeric solids-concentration profiles** for one
or more of the six batch experiments (Destelbergen Co = 2.40 / 3.23 / 4.30 g/L;
Deinze Co = 3.67 / 6.12 / 7.29 g/L)? Any convenient format works — ideally either:

- a CSV/matrix of `C` (g/L) with **rows = measurement times** and **columns =
  heights z**, plus the height and time vectors; or
- a tidy table with columns `t (min), z (m), C (g/L)`.

Together with the **column height**, **initial concentration Co**, and the
**spatial/temporal sampling resolution**, that is everything I need.

The data would be used solely to validate the method on real measurements. I will
of course cite De Clercq et al. (2005) and the thesis, and gratefully acknowledge
you for sharing the data (and I'm happy to discuss co-authorship if that is more
appropriate given the contribution).

Thank you very much for considering this.

Best regards,
Davor Runje
PhD student, Faculty of Electrical Engineering and Computing (FER), University of Zagreb
Code / method: https://github.com/davorrunje/mononet

---

*Notes for us:* the ideal deliverable is a time-resolved `C(z,t)` field (many
depths x many times) for at least one experiment — that is what enables the
sparse-detector inverse reconstruction. A single-time snapshot or a settling
curve `h(t)` alone is not sufficient. If they can only share one experiment,
Co = 3.23 g/L (Destelbergen, the one plotted in Fig 5.8) or any with a clear
descending interface is a good choice.
