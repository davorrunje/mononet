# Relicense to Apache-2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relicense `mononet` from PolyForm Noncommercial 1.0.0 to Apache-2.0 across all license-bearing files, with SPDX headers on first-party Python.

**Architecture:** Documentation/metadata only — no source-code logic changes. Five reviewable units: (1) `LICENSE` + `NOTICE.md`; (2) `pyproject.toml`; (3) user-facing prose docs; (4) agent/meta docs; (5) SPDX headers. Verification is grep assertions plus build/lint/test staying green.

**Tech Stack:** Apache License 2.0, SPDX identifiers, `uv` (build/lint/test), ruff, mypy, Sphinx.

## Global Constraints

- Target license: **Apache License 2.0**, SPDX `Apache-2.0`.
- Copyright holder: **AIRT Technologies Ltd.**, years **2023-2026**.
- Patent posture: rely on Apache-2.0 section 3's built-in grant for use of the code. `NOTICE.md` describes US Patent 11,551,063 **factually** — no "reserved", no "separate license required", no commercial-licensing contact, no `licensing@airt.ai`.
- Contributions: inbound=outbound under Apache-2.0 section 5. No CLA, no DCO.
- SPDX header: the exact line `# SPDX-License-Identifier: Apache-2.0`, placed as the first line of each first-party `.py` file (after a shebang if present, above the module docstring).
- No source-code behavior, public API, or release-pipeline changes.
- Historical dated specs/plans under `docs/superpowers/` are left as record, EXCEPT the meta-spec gets a dated superseding note (Task 4).
- Branch: `chore/relicense-apache-2.0` (already created, holds the design spec). Never commit to `main`. Commits signed (Secretive SSH); end messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- All commands run from repo root `/Users/davor/Projects/PhD/mononet`.

---

### Task 1: Replace `LICENSE` and rewrite `NOTICE.md`

**Files:**
- Modify (full replace): `LICENSE`
- Modify (full replace): `NOTICE.md`

**Interfaces:**
- Consumes: nothing.
- Produces: the canonical Apache-2.0 `LICENSE` and a lean Apache NOTICE that Task 2's `license-files` bundles and Task 3/4 prose reference.

- [ ] **Step 1: Replace `LICENSE` with the verbatim Apache-2.0 text**

Obtain the canonical Apache License 2.0 plain text from
<https://www.apache.org/licenses/LICENSE-2.0.txt> (e.g.
`curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE`).
If the environment has no network access, STOP and report NEEDS_CONTEXT — do
not hand-retype the license body.

Then prepend the copyright line as the first two lines of the file, above the
`                                 Apache License` header:

```
Copyright 2023-2026 AIRT Technologies Ltd.

                                 Apache License
                           Version 2.0, January 2004
```

The body below `Apache License` must be the unmodified canonical text
(through the end of the APPENDIX). Do not paraphrase.

- [ ] **Step 2: Verify `LICENSE` content**

Run:

```bash
head -1 LICENSE
grep -c "Apache License" LICENSE
grep -c "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" LICENSE
grep -ci "polyform" LICENSE
```

Expected: line 1 is `Copyright 2023-2026 AIRT Technologies Ltd.`; "Apache License" count >= 1; the TERMS heading count is 1; PolyForm count is 0.

- [ ] **Step 3: Rewrite `NOTICE.md`**

Replace the entire contents of `NOTICE.md` with:

```markdown
# NOTICE

mononet
Copyright 2023-2026 AIRT Technologies Ltd.

This product is licensed under the Apache License, Version 2.0 (see
`LICENSE`).

## Patent

This software implements technology described in U.S. Patent No.
11,551,063 ("Implementing monotonic constrained neural networks",
assignee: AIRT Technologies Ltd.,
<https://patents.justia.com/patent/11551063>).

The software is distributed under the Apache License 2.0. Section 3 of
that license grants you the patent rights needed to use, make, and
distribute this software and your derivative works of it.

## Reference paper

Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic Neural
Networks.* ICML 2023. <https://arxiv.org/abs/2205.11775>

If you use `mononet` in academic work, please cite this paper.

## Trademarks

PyTorch is a trademark of the Linux Foundation. JAX is a trademark of
Google LLC. Keras is a trademark of Google LLC. Use of these trademark
names here indicates compatibility, not endorsement.
```

- [ ] **Step 4: Verify `NOTICE.md`**

Run:

```bash
grep -ci "polyform\|noncommercial\|reserved\|licensing@airt\|separate patent license\|does \*\*not\*\* grant" NOTICE.md
grep -c "Apache License" NOTICE.md
```

Expected: first count `0` (no reservation/noncommercial/commercial-contact language remains); "Apache License" count >= 1.

- [ ] **Step 5: Commit**

```bash
git add LICENSE NOTICE.md
git commit -m "$(cat <<'EOF'
license: replace PolyForm with Apache-2.0; rewrite NOTICE factually

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Update `pyproject.toml` license metadata

**Files:**
- Modify: `pyproject.toml:13` (license field), `pyproject.toml:30-37` (`[project.urls]`, remove Patent)

**Interfaces:**
- Consumes: `LICENSE`, `NOTICE.md` from Task 1 (referenced by `license-files`).
- Produces: package metadata declaring `Apache-2.0`.

- [ ] **Step 1: Change the license field**

In `pyproject.toml`, replace line 13:

```toml
license = "LicenseRef-PolyForm-Noncommercial-1.0.0"
```

with:

```toml
license = "Apache-2.0"
```

Leave line 14 (`license-files = ["LICENSE", "NOTICE.md"]`) unchanged.

- [ ] **Step 2: Remove the Patent URL**

In the `[project.urls]` table, delete this line entirely:

```toml
Patent        = "https://patents.justia.com/patent/11551063"
```

Leave the other URL entries (Homepage, Documentation, Repository, Issues,
Changelog, Paper) unchanged.

- [ ] **Step 3: Verify metadata and a clean build**

Run:

```bash
grep -n "^license = " pyproject.toml
grep -ci "polyform\|11551063" pyproject.toml
uv build
```

Expected: the license line reads `license = "Apache-2.0"`; the second grep count is `0`; `uv build` succeeds.

Then confirm the built wheel's metadata records the new license:

```bash
unzip -p dist/mononet-*-py3-none-any.whl '*.dist-info/METADATA' | grep -i "^License"
```

Expected: the METADATA shows `License-Expression: Apache-2.0` (hatchling emits
the SPDX expression). (`dist/` is git-ignored — do not commit it.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
build: declare Apache-2.0 license; drop patent URL from project metadata

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Rewrite user-facing license prose

**Files:**
- Modify: `README.md` (the `## License & patent` section)
- Modify (full replace): `docs/about/license.md`
- Modify: `CONTRIBUTING.md` (the `## License & patent reminder` section, lines ~6-13)
- Modify: `CHANGELOG.md` (add a relicense entry under `## [Unreleased]`)

**Interfaces:**
- Consumes: `LICENSE`, `NOTICE.md` from Task 1.
- Produces: user-facing docs consistent with Apache-2.0.

- [ ] **Step 1: Rewrite the `README.md` license section**

Replace the `## License & patent` section (currently three lines beginning
"Code: PolyForm Noncommercial 1.0.0...") with:

```markdown
## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
Commercial use is permitted. The technique is described in U.S. Patent
11,551,063 (assignee: AIRT Technologies Ltd.); the Apache-2.0 license
grants the patent rights needed to use this code. For academic use, please
cite the paper (see [`NOTICE.md`](NOTICE.md)).
```

(Keep the heading text reachable from any existing anchor: the section is
renamed `## License`. Leave the surrounding `## Formal proofs` section
intact.)

- [ ] **Step 2: Rewrite `docs/about/license.md`**

Replace the entire contents of `docs/about/license.md` with:

```markdown
# License

## Code license

`mononet` is licensed under the **Apache License 2.0**. See the
[full text](https://www.apache.org/licenses/LICENSE-2.0) or the `LICENSE`
file in the repository. Commercial and noncommercial use, modification, and
redistribution are permitted under the license's terms, including its
attribution and NOTICE requirements.

## Patent

The technique implemented by `mononet` is described in **U.S. Patent No.
11,551,063** ("Implementing monotonic constrained neural networks",
assignee: **AIRT Technologies Ltd.**, Zagreb, Croatia,
<https://patents.justia.com/patent/11551063>).

`mononet` is distributed under the Apache License 2.0, whose section 3
grants the patent rights needed to use, make, and distribute this software.

## Contributions

Contributions are accepted under the Apache License 2.0 (inbound=outbound,
per the license's section 5). No separate contributor agreement is
required.
```

- [ ] **Step 3: Rewrite the `CONTRIBUTING.md` license section**

Replace the `## License & patent reminder` section (lines ~6-13, beginning
"`mononet` is distributed under the PolyForm...") with:

```markdown
## License of contributions

`mononet` is licensed under the Apache License 2.0. Contributions are
accepted inbound=outbound under section 5 of that license: unless you state
otherwise, any contribution you intentionally submit for inclusion is
licensed under Apache-2.0, with no additional terms. No CLA is required. See
[`NOTICE.md`](NOTICE.md).
```

- [ ] **Step 4: Add a `CHANGELOG.md` relicense entry**

Under the `## [Unreleased]` heading, add a new `### Changed` subsection (place
it after the existing `### Added` block; create `### Changed` if absent):

```markdown
### Changed
- Relicensed from PolyForm Noncommercial License 1.0.0 to the **Apache
  License 2.0**, following AIRT Technologies Ltd.'s decision to discontinue
  patent-related activities. Apache-2.0's section 3 grants the patent
  rights needed to use the code. Effective from the first PyPI release.
```

- [ ] **Step 5: Verify**

Run:

```bash
grep -ci "polyform\|noncommercial\|licensing@airt" README.md docs/about/license.md CONTRIBUTING.md
grep -c "Apache" README.md docs/about/license.md CONTRIBUTING.md CHANGELOG.md
```

Expected: the first command reports `0` matches in every listed file (count
lines all `:0`); the second shows `Apache` present (count >= 1) in each.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/about/license.md CONTRIBUTING.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs: rewrite user-facing license prose for Apache-2.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Update agent/meta docs

**Files:**
- Modify: `CLAUDE.md:8` (the license repo-posture bullet)
- Modify: `docs/superpowers/specs/2026-05-21-mononet-package-design.md` (add a dated superseding note to its license section, §7)

**Interfaces:**
- Consumes: the decisions in this plan's Global Constraints.
- Produces: agent guidance and the meta-spec consistent with Apache-2.0.

- [ ] **Step 1: Rewrite the `CLAUDE.md` license posture bullet**

Replace the bullet at `CLAUDE.md:8` (beginning "**License is PolyForm
Noncommercial 1.0.0**...") with:

```markdown
- **License is Apache-2.0** (assignee/copyright holder: AIRT Technologies Ltd.). Commercial and noncommercial use are both permitted. The underlying technique is described in **U.S. Patent 11,551,063**; the Apache-2.0 license (section 3) grants the patent rights needed to use this code, and AIRT does not pursue patent-related activities. There is no noncommercial restriction. See [NOTICE.md](NOTICE.md).
```

Leave the other "Repo posture" bullets (tone guidance, etc.) unchanged.

- [ ] **Step 2: Add a dated superseding note to the meta-spec license section**

In `docs/superpowers/specs/2026-05-21-mononet-package-design.md`, locate the
license section heading `## 7. License, NOTICE, and documentation`. Insert a
blockquote note immediately below that heading (do not rewrite the historical
body):

```markdown
> **Superseded 2026-06-28:** the project relicensed from PolyForm
> Noncommercial 1.0.0 to **Apache-2.0**. The PolyForm details below are
> retained as historical record; the current license posture is defined in
> [`2026-06-28-relicense-apache-2.0-design.md`](2026-06-28-relicense-apache-2.0-design.md).
```

- [ ] **Step 3: Verify**

Run:

```bash
grep -ci "polyform\|noncommercial\|licensing@airt\|hard constraint" CLAUDE.md
grep -c "Superseded 2026-06-28" docs/superpowers/specs/2026-05-21-mononet-package-design.md
```

Expected: the `CLAUDE.md` count is `0`; the meta-spec note count is `1`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-05-21-mononet-package-design.md
git commit -m "$(cat <<'EOF'
docs: update agent posture and meta-spec for Apache-2.0 relicense

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Add SPDX headers to first-party Python files

**Files:**
- Modify: every `.py` under `mononet/` (15), `tests/` (25), `tools/` (3), and `docs/` excluding `_build/` (2) — 45 files total.

**Interfaces:**
- Consumes: the SPDX identifier `Apache-2.0`.
- Produces: machine-readable per-file license markers.

- [ ] **Step 1: Apply the SPDX header idempotently**

Run this script (handles shebangs, skips files that already carry the header,
inserts the line above the module docstring):

```bash
python3 - <<'PY'
import pathlib
HEADER = "# SPDX-License-Identifier: Apache-2.0\n"
roots = ["mononet", "tests", "tools", "docs"]
changed, skipped = [], []
for root in roots:
    for p in pathlib.Path(root).rglob("*.py"):
        if "_build" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        if "SPDX-License-Identifier" in text:
            skipped.append(str(p)); continue
        lines = text.splitlines(keepends=True)
        insert_at = 1 if (lines and lines[0].startswith("#!")) else 0
        lines.insert(insert_at, HEADER)
        p.write_text("".join(lines), encoding="utf-8")
        changed.append(str(p))
print(f"changed={len(changed)} skipped={len(skipped)}")
PY
```

Expected output: `changed=45 skipped=0` (if a rerun: `changed=0 skipped=45`).

- [ ] **Step 2: Verify coverage and placement**

Run:

```bash
echo "files: $(find mononet tests tools docs -name '*.py' -not -path '*/_build/*' | wc -l | tr -d ' ')"
echo "with header: $(grep -rl 'SPDX-License-Identifier: Apache-2.0' mononet tests tools docs --include='*.py' | grep -v '/_build/' | wc -l | tr -d ' ')"
head -2 mononet/core/types.py
head -3 mononet/torch/layers.py
```

Expected: `files` and `with header` are both `45`; `mononet/core/types.py`
line 1 is `# SPDX-License-Identifier: Apache-2.0` and line 2 is the docstring
opener `"""Shared types used by all mononet backends.`; `mononet/torch/layers.py`
shows the SPDX line, then the docstring, then `from __future__ import annotations`.

- [ ] **Step 3: Confirm lint, types, and tests stay green**

Run:

```bash
uv run ruff check --exit-non-zero-on-fix
uv run ruff format --check
MONONET_TEST_BACKEND=torch uv run pytest -q tests/core tests/torch tests/equivalence tests/test_top_level_imports.py
```

Expected: ruff check passes (no errors, nothing to fix); `ruff format --check`
reports all files already formatted; pytest passes (70 passed). The leading
comment must not trigger any docstring (`D`) or import-order (`I`) findings —
if it does, STOP and report (do not suppress rules to force green).

- [ ] **Step 4: Confirm the docs still build strict**

Run: `./tools/build-docs.sh`
Expected: exit 0 (the two `docs/*.py` files gaining a header must not break
the Sphinx build).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: add SPDX-License-Identifier headers to first-party Python

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the executor

- This is a license/docs migration — no application logic changes. The
  "tests" are grep assertions and keeping build/lint/test green.
- The PolyForm text deliberately survives in historical dated specs/plans
  under `docs/superpowers/` (e.g. the lean-proofs and injective-flows specs,
  the scaffold-migration plan). A final-acceptance `grep -ri polyform .`
  returning only those dated files (plus this plan/spec, which name PolyForm
  to describe the change) is expected and correct.
- `./tools/get-version.sh` uses PCRE grep and fails on macOS; use
  `uv version --short` locally if you need the version. Not relevant to this
  plan's edits.
- Do not commit `dist/` (git-ignored) produced by `uv build` in Task 2.
