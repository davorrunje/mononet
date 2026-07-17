# scholar — Visual Identity (v1, text)

**Date:** 2026-07-17 · **Status:** first proposal (text/SVG). A browser-companion
pass (`/design consent`) can later render mockups; this establishes the direction.

`scholar` is the *scientific* counterpart to `superpowers`. Its identity should
read **scholarly, rigorous, and calm** — a trusted mentor/colleague, not a flashy
product. Lowercase, understated, typographic; the opposite of hype.

## Name & wordmark

- **Name:** `scholar` (always lowercase, like `superpowers`).
- **Wordmark:** the word set in a humanist **serif** (scholarship, print, the
  paper), lowercase. Optional bracket motif around it — `[ scholar ]` — nodding to
  citations. See `assets/wordmark.svg`.
- **Skill names** render in **monospace** (`grill`, `literature`, `hypothesis-
  testing`) — they are commands.

## Logo mark

Recommended concept — **"citation & through-line"** (`assets/logo-concept.svg`):
two citation brackets `[ ]` embracing an upward stroke with a dot above it. Reads
as: *scholarship (brackets/citations) enclosing a claim (dot) and its rising
through-line (understanding/argument)*. Abstract, geometric, works at favicon size,
pairs with a code aesthetic.

Alternatives considered (not chosen for v1):
- **Owl** — the classic scholar/wisdom symbol; warm but common and harder to keep
  distinctive at small sizes.
- **Mortarboard** — too narrowly "graduation/degree"; undersells the research-
  program and colleague framing.
- **Compass/through-line** — nice for "coherent narrative" but less legible as a
  mark.

## Color palette

Academic and quiet, with one warm accent for *insight*.

| Role | Name | Hex | Use |
|---|---|---|---|
| Primary text / dark | Ink | `#16181D` | body text, dark UI |
| Brand | Indigo | `#4338CA` | logo brackets, links, primary accents |
| Accent | Amber | `#F59E0B` | the "insight"/through-line, highlights (sparingly) |
| Positive | Scholar Green | `#059669` | **"refuted = done"** states, passed checks |
| Secondary text | Slate | `#4B5563` | captions, metadata |
| Light background | Parchment | `#FAF7F0` | page background, cards |

Note the deliberate choice to give **refuted/negative-result** states a *positive*
green — reinforcing the principle that a refuted hypothesis is successful science
(anti-Goodhart, ADR-0014).

## Typography

- **Display / headings:** a scholarly serif — **Source Serif 4** (or Charter /
  Lora). Evokes the paper.
- **Body / UI:** a humanist sans — **Inter** (or IBM Plex Sans).
- **Code / skill names / registries:** **JetBrains Mono** (or IBM Plex Mono).

Pairing rationale: serif headings signal *scholarship*; sans body keeps docs
readable; mono marks the *machine* parts (skills, YAML, run-refs).

## Voice & tone

Mentor-and-colleague: rigorous but supportive; asks more than it asserts; never
grandiose. Directly reflects the mentor personas (ADR-0016) and the two
principles — it advises and grills, but *you* decide and *you* author.

- **Primary tagline:** *Research you can defend.*
- **Secondary lines:** *You drive.* · *Understand your work.*

## Usage notes

- Prefer the wordmark alone in docs; use the mark for favicon/social/avatar.
- Amber is an accent, not a fill — one highlight per view.
- Keep whitespace generous; the identity is calm, not busy.
- On dark backgrounds, Parchment text on Ink; Indigo lightens to `#818CF8`.

## Assets

- `assets/wordmark.svg` — the lowercase serif wordmark with bracket motif.
- `assets/logo-concept.svg` — the citation & through-line mark.

*(These are v1 concepts to react to; a browser-companion pass can iterate on real
mockups, favicons, and social cards.)*
