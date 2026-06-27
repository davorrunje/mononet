# Relicense to Apache-2.0 — Design

**Date:** 2026-06-28
**Status:** Approved
**Scope:** Relicense `mononet` from PolyForm Noncommercial License 1.0.0 to the
Apache License 2.0, reflecting AIRT Technologies Ltd.'s decision to stop
pursuing patent-related activities. Documentation and metadata only — no
source-code logic changes.

> **Not legal advice.** AIRT owns both the copyright and US Patent 11,551,063
> and is therefore the relicensing authority. This spec records an
> already-made company decision; AIRT's counsel should review the final
> `LICENSE` and `NOTICE.md` text before the first Apache-licensed release.

## 1. Decisions

| Topic | Decision |
|---|---|
| License | **Apache License 2.0** (SPDX `Apache-2.0`). |
| Patent | Rely on Apache-2.0's built-in patent grant (section 3) for use of the code. AIRT retains the patent formally; `NOTICE.md` describes it factually, with no reservation or non-assertion language. No formal dedication/abandonment. |
| Contributions | Inbound = outbound under Apache-2.0 section 5. No CLA, no DCO. |
| Source headers | Add the one-line `# SPDX-License-Identifier: Apache-2.0` to every first-party Python file. |
| Patent URL | Remove `Patent = ...` from `pyproject.toml [project.urls]`. |
| Copyright holder | `AIRT Technologies Ltd.`, years `2023-2026`. |

## 2. Why Apache-2.0

It is the only mainstream license that is simultaneously permissive
(commercial use allowed), carries an **express patent grant** that
operationalizes the "stop patent activity" decision for users of this code,
and matches two of the three backends (JAX and Keras are Apache-2.0; PyTorch
and numpy are BSD). MIT/BSD grant only an implied patent license — undesirable
ambiguity when the distributor *holds* the patent.

## 3. Background — what changes

The PolyForm license + patent reservation is currently referenced across:
`LICENSE`, `NOTICE.md`, `pyproject.toml`, `README.md`, `CHANGELOG.md`,
`CONTRIBUTING.md`, `CLAUDE.md`, `docs/about/license.md`, and several dated
design docs under `docs/superpowers/`. There are **no per-file license
headers** today (only `pyproject.toml` carries the SPDX string).

`mononet 0.0.0a0` was published only to **TestPyPI** (ephemeral,
non-authoritative) under PolyForm. No authoritative PolyForm distribution
exists to reconcile; the first real PyPI release carries Apache-2.0.

## 4. Changes

### 4.1 `LICENSE`

Replace the entire PolyForm text with the verbatim Apache License 2.0 text.
Prepend the copyright line above the license body:

```
Copyright 2023-2026 AIRT Technologies Ltd.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
...
```

Use the canonical Apache-2.0 text from <https://www.apache.org/licenses/LICENSE-2.0.txt>.

### 4.2 `NOTICE.md`

Rewrite as an Apache-2.0 NOTICE file. Required content:

- **Attribution:** `mononet` — Copyright 2023-2026 AIRT Technologies Ltd.,
  licensed under the Apache License 2.0 (see `LICENSE`).
- **Patent (factual, non-reserving):** the software implements technology
  covered by US Patent No. 11,551,063 ("Implementing monotonic constrained
  neural networks", assignee AIRT Technologies Ltd.); the code is distributed
  under the Apache License 2.0, whose section 3 grants the patent rights
  needed to use, make, and distribute this software. No statement that the
  patent is "reserved", that a separate license is "required", or any
  commercial-licensing contact.
- **Citation:** keep the reference-paper / BibTeX pointer
  (Runje & Shankaranarayana, ICML 2023).
- **Trademarks:** keep the existing PyTorch/JAX/Keras trademark note.

Remove the `licensing@airt.ai` "commercial licensing" block entirely.

Keep `NOTICE.md` lean: under Apache-2.0 section 4(d), downstream redistributors
must propagate NOTICE contents, so it should carry only attribution, the
factual patent note, citation, and trademarks.

### 4.3 `pyproject.toml`

- `license = "LicenseRef-PolyForm-Noncommercial-1.0.0"` → `license = "Apache-2.0"`.
- Keep `license-files = ["LICENSE", "NOTICE.md"]`.
- Remove the `Patent = "https://patents.justia.com/patent/11551063"` entry from
  `[project.urls]`.
- No `License ::` trove classifier exists, so none to remove.

### 4.4 `README.md`

Rewrite the `## License & patent` section to a short Apache-2.0 statement:
the code is Apache-2.0 (see `LICENSE` / `NOTICE.md`); cite the paper for
academic use. Drop the "reserved" / "commercial licensing" framing and the
`licensing@airt.ai` contact.

### 4.5 `docs/about/license.md`

Rewrite to describe Apache-2.0 (mirroring the README statement) and point to
`LICENSE` / `NOTICE.md`. Remove PolyForm/noncommercial and patent-reservation
text.

### 4.6 `CONTRIBUTING.md`

Remove any noncommercial framing. Add a one-line contribution-licensing note:
contributions are accepted inbound=outbound under Apache-2.0 section 5 (no CLA
or DCO required).

### 4.7 `CHANGELOG.md`

Add an entry recording the relicense: PolyForm Noncommercial License 1.0.0 →
Apache License 2.0, with a one-line rationale (AIRT decision to discontinue
patent-related activities). Note the change applies from the first real PyPI
release onward.

### 4.8 `CLAUDE.md`

Rewrite the **Repo posture** bullet on licensing. Remove:
- "License is PolyForm Noncommercial 1.0.0" and the noncommercial hard-constraint.
- "do not propose features whose primary purpose is helping commercial deployments".
- "do not suggest copying ourselves into permissively-licensed repos".
- "route any 'can I use this commercially?' question to licensing@airt.ai".

Replace with: the project is licensed under **Apache-2.0**; commercial use is
permitted; the underlying patent (US 11,551,063) is granted for use of this
code via Apache section 3; AIRT does not pursue patent-related activities.
Keep the senior-collaborator tone guidance.

### 4.9 `docs/superpowers/specs/2026-05-21-mononet-package-design.md`

Update only the "license posture" subsection (CLAUDE.md cites this meta-spec as
current truth) to reflect Apache-2.0. Add a dated note that the project
relicensed on 2026-06-28. Leave all other dated specs/plans unchanged as
historical record.

### 4.10 SPDX headers on first-party Python

Add this exact line as the **first line** of every first-party `.py` file,
immediately above the existing module docstring (a comment is not a statement,
so the docstring remains the module's first statement):

```python
# SPDX-License-Identifier: Apache-2.0
```

Scope: all `.py` under `mononet/` (15 files, the only files shipped in the
wheel), plus `tests/` (25), `tools/` (3), and `docs/` non-build (2) for
uniformity — 45 files total. Files that already begin with
`from __future__ import annotations` keep the SPDX line above the docstring;
the `__future__` import stays the first executable statement.

## 5. Acceptance

- `LICENSE` is the verbatim Apache-2.0 text with the AIRT copyright line; no
  PolyForm text remains anywhere (`grep -ri polyform .` returns only historical
  dated specs/plans deliberately left as record).
- `NOTICE.md` is a lean Apache NOTICE with a factual (non-reserving) patent
  note and no `licensing@airt.ai` block.
- `pyproject.toml` reports `license = "Apache-2.0"`; no Patent URL; `uv build`
  succeeds and the built wheel's metadata shows the Apache-2.0 license.
- Every first-party `.py` file starts with the SPDX one-liner; the test suite
  still passes and ruff/mypy are clean (the comment line must not trip
  docstring or import-order rules).
- `README.md`, `docs/about/license.md`, `CONTRIBUTING.md`, `CLAUDE.md`, and the
  meta-spec posture section describe Apache-2.0 with no noncommercial or
  patent-reservation language.
- Strict docs build (`./tools/build-docs.sh`) passes.

## 6. Consequences (recorded for posterity)

- Commercial use of the code becomes permitted; the PolyForm noncommercial
  monetization path is closed. Intended.
- Relicensing is irrevocable for any version distributed under Apache-2.0;
  future versions may carry a different license but distributed copies stay
  Apache-2.0.
- The Apache-2.0 patent grant covers the claims practiced by the distributed
  code. It does not license a clean-room reimplementation of the patent; AIRT
  has chosen not to formally dedicate or abandon the patent.
- Contributions flow in under Apache section 5; no contributor agreements to
  administer.

## 7. Non-goals

- No formal patent dedication, disclaimer, or abandonment filing.
- No change to source-code behavior, the public API, or the release pipeline.
- No rewrite of historical dated specs/plans beyond the meta-spec posture note.
- No CLA/DCO infrastructure.
