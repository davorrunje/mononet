# defendable-science migration — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move this repository from the `honest-scholar` plugin/CLI to its renamed successor `defendable-science`, so the plugin loads, the CLI reads its configuration, and no stale name remains outside immutable git history.

**Architecture:** Three separable changes. First the *live wiring* — plugin id, marketplace ref, and the config directory the CLI actually reads. Then a *mechanical name sweep* over prose and research records. Then the one *non-mechanical* edit: `CLAUDE.md`'s dependency rationale, which this migration reverses. Verification hinges on one check that fails loudly, because the failure mode here is silent.

**Tech Stack:** Claude Code plugin marketplace (`.claude/settings.json`), `defendable-science` 0.2.1 (PyPI, `dev` group), `uv`, `git mv`, `sed`, pre-commit.

**Spec:** [`docs/superpowers/specs/2026-08-26-defendable-science-migration-design.md`](../specs/2026-08-26-defendable-science-migration-design.md)

## Global Constraints

- Upstream renamed everything in `0.2.0` with **no deprecation shim**; the `honest-scholar` PyPI distribution is abandoned at `0.1.1`. There is nothing to fall back to.
- Exact renames: `honest-scholar` → `defendable-science`; `honest_scholar` → `defendable_science`; `.honest-scholar/` → `.defendable-science/`; `honest-scholar.science` → `defendable.science`; `davorrunje/honest-scholar` → `davorrunje/defendable-science`; `HonestScholar-Skill:` → `DefendableScience-Skill:`; CLI alias `hsch` → `dsci`; env vars `HONEST_SCHOLAR_KEYS_PATH` → `DEFENDABLE_SCIENCE_KEYS_PATH`, `HONEST_SCHOLAR_LIVE` → `DEFENDABLE_SCIENCE_LIVE` (neither env var appears in this repo).
- Marketplace ref is pinned to **`v0.2.1`**, matching the `defendable-science==0.2.1` CLI pin already in the `dev` group. Not `v0.2.0`, which upstream's install snippet shows but is older than the current release.
- **Two files must keep the old name and must be excluded from every sweep:**
  `docs/superpowers/specs/2026-08-26-defendable-science-migration-design.md` and
  `docs/superpowers/plans/2026-08-26-defendable-science-migration.md` (this plan).
  Both document the rename and contain "before" columns; rewriting them destroys their meaning. This is the single easiest way to silently ruin this migration.
- The existing commits carrying `HonestScholar-Skill:` are immutable — two reachable from `origin/main` (eight across all refs, including unmerged branches). History keeps the old name; that is correct, not a gap.
- Commit on the current branch `chore/defendable-science-migration`, never `main`. Use `git commit --no-gpg-sign`, ending messages with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Do not run `uv sync` — the project `.venv` is a torch-GPU environment that is slow to rebuild. `defendable-science==0.2.1` is already installed.

## File Structure

| File | Change |
|---|---|
| `.claude/settings.json` | **modify** — plugin id, marketplace key, repo, ref `v0.1.0` → `v0.2.1` |
| `.honest-scholar/` → `.defendable-science/` | **git mv** — `config.yml`, `rclone.conf.example` |
| `.gitignore` | **modify** — 6 lines under the workflow comment |
| `docs/superpowers/specs/2026-07-21-honest-scholar-integration-design.md` | **git mv** + content sweep → `…-defendable-science-integration-design.md` |
| `CLAUDE.md` | **modify** — sweep, plus a hand-written rationale replacement (Task 3) |
| `datasets.yml`, `.claude/skills/create-pr/STYLE.md` | **modify** — sweep |
| `docs/research/**` (21 files) | **modify** — sweep |
| `docs/superpowers/specs/2026-08-26-defendable-science-migration-design.md` | **excluded from sweeps** |
| `docs/superpowers/plans/2026-08-26-defendable-science-migration.md` | **excluded from sweeps** |

---

### Task 1: Live wiring — plugin, config directory, gitignore

**Files:**
- Modify: `.claude/settings.json`
- Rename: `.honest-scholar/` → `.defendable-science/`
- Modify: `.gitignore` (the 6-line block beginning `# honest-scholar workflow (research-init adopt)`)

**Interfaces:**
- Consumes: nothing.
- Produces: `.defendable-science/config.yml` at the path `defendable_science.core.config.DEFAULT_CONFIG_PATH` points to. Later tasks only rewrite text; nothing depends on their output.

- [ ] **Step 1: Confirm the failure state before changing anything**

This is the check that will prove the migration worked, so establish that it currently fails:

```bash
uv run --no-sync python -c "from defendable_science.core.config import load_config; print(sorted(load_config()))"
```

Expected: `[]` — the new CLI finds no config, because `load_config` returns `{}` for a missing file rather than erroring. That silence is the trap this task closes.

- [ ] **Step 2: Rename the config directory**

```bash
git mv .honest-scholar .defendable-science
ls .defendable-science/
```

Expected: `config.yml  rclone.conf.example`. Use `git mv`, not `mv`, so the rename is recorded rather than appearing as a delete plus an add.

- [ ] **Step 3: Verify the CLI now reads the config**

```bash
uv run --no-sync python -c "from defendable_science.core.config import load_config; c = load_config(); print(sorted(c)); assert c, 'config not found at the default path'"
```

Expected: `['engineering_backend', 'experiment_backend', 'literature', 'mirror']` and no assertion error. If this still prints `[]`, stop — the rest of the plan is cosmetic and this is the part that matters.

- [ ] **Step 4: Rewrite the plugin wiring**

Replace the whole of `.claude/settings.json` with:

```json
{
  "enabledPlugins": {
    "superpowers@claude-plugins-official": false,
    "superpowers@superpowers-dev": true,
    "defendable-science@defendable-science": true
  },
  "extraKnownMarketplaces": {
    "superpowers-dev": {
      "source": {
        "source": "git",
        "url": "https://github.com/obra/superpowers.git"
      }
    },
    "defendable-science": {
      "source": {
        "source": "github",
        "repo": "davorrunje/defendable-science",
        "ref": "v0.2.1"
      }
    }
  }
}
```

- [ ] **Step 5: Verify the settings file and the pinned tag**

```bash
python3 -c "import json; d=json.load(open('.claude/settings.json')); print(d['enabledPlugins']); print(d['extraKnownMarketplaces']['defendable-science'])"
gh api repos/davorrunje/defendable-science/tags --jq '.[].name' | grep -x v0.2.1
```

Expected: valid JSON showing `defendable-science@defendable-science: True`, the marketplace pinned to `v0.2.1`, and `grep` echoing `v0.2.1` — proving the pinned tag exists upstream rather than assuming it.

- [ ] **Step 6: Update `.gitignore`**

Replace the block:

```
# honest-scholar workflow (research-init adopt)
.datasets-cache/
.honest-scholar/rclone.conf
# local key store (secrets) — never commit (honest-scholar#66)
.honest-scholar/keys.json
# content-addressed Tier-B dataset cache (honest-scholar `dataset fetch|verify`)
.honest-scholar/cache/
```

with:

```
# defendable-science workflow (research-init adopt)
.datasets-cache/
.defendable-science/rclone.conf
# local key store (secrets) — never commit (defendable-science#66)
.defendable-science/keys.json
# content-addressed Tier-B dataset cache (defendable-science `dataset fetch|verify`)
.defendable-science/cache/
```

- [ ] **Step 7: Confirm the ignore rules still bite**

```bash
touch .defendable-science/keys.json
git status --porcelain .defendable-science/
git check-ignore -v .defendable-science/keys.json
rm .defendable-science/keys.json
```

Expected: `git status` shows **nothing** for the ignored file (only tracked changes, if any), and `check-ignore` names the `.gitignore` line. A secrets file that stops being ignored is the worst possible outcome of this task, so prove it rather than assume it.

- [ ] **Step 8: Commit**

```bash
# NOT `git add -A ... .honest-scholar`: that path no longer exists after the
# `git mv`, and the failed pathspec aborts the whole `git add`, staging nothing.
git add -A .claude/settings.json .gitignore .defendable-science
git diff --cached --stat   # confirm all four files are staged
git commit --no-gpg-sign -m "$(cat <<'MSG'
chore: point the plugin, config dir and ignores at defendable-science

honest-scholar was renamed in its 0.2.0 release with no deprecation
shim. The CLI reads .defendable-science/config.yml
(DEFAULT_CONFIG_PATH), and load_config returns {} for a missing file --
so before this commit every binding in config.yml was being silently
ignored rather than erroring.

Marketplace pinned to v0.2.1 to match the defendable-science==0.2.1 CLI
pin in the dev group, so plugin and CLI move together.

Verified: load_config() now returns the four top-level keys where it
returned [] before, and .defendable-science/keys.json is still ignored.

Refs #-migration

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

Replace `Refs #-migration` with the issue number if one exists; otherwise delete that line.

---

### Task 2: Mechanical name sweep

**Files:**
- Rename: `docs/superpowers/specs/2026-07-21-honest-scholar-integration-design.md` → `docs/superpowers/specs/2026-07-21-defendable-science-integration-design.md`
- Modify: `CLAUDE.md`, `datasets.yml`, `.claude/skills/create-pr/STYLE.md`, `.defendable-science/config.yml`, `.defendable-science/rclone.conf.example`, the renamed spec, and 21 files under `docs/research/`
- **Excluded:** `docs/superpowers/specs/2026-08-26-defendable-science-migration-design.md`, `docs/superpowers/plans/2026-08-26-defendable-science-migration.md`

**Interfaces:**
- Consumes: the renamed `.defendable-science/` directory from Task 1.
- Produces: nothing later tasks read. Task 3 rewrites two `CLAUDE.md` paragraphs this task will have already swept.

- [ ] **Step 1: Rename the dated spec**

```bash
git mv docs/superpowers/specs/2026-07-21-honest-scholar-integration-design.md \
       docs/superpowers/specs/2026-07-21-defendable-science-integration-design.md
```

The date stays; only the name changes.

- [ ] **Step 2: Run the sweep**

Order matters: `honest-scholar.science` must be rewritten **before** the bare `honest-scholar`, or it becomes `defendable-science.science`.

```bash
files=$(git ls-files -z \
  | xargs -0 grep -lE 'honest.scholar|honest_scholar|HonestScholar|HONEST_SCHOLAR' 2>/dev/null \
  | grep -v '2026-08-26-defendable-science-migration')

echo "$files" | tr ' ' '\n'   # review the list before editing

for f in $files; do
  sed -i \
    -e 's/honest-scholar\.science/defendable.science/g' \
    -e 's/HonestScholar/DefendableScience/g' \
    -e 's/HONEST_SCHOLAR/DEFENDABLE_SCIENCE/g' \
    -e 's/honest_scholar/defendable_science/g' \
    -e 's/honest-scholar/defendable-science/g' \
    "$f"
done
```

- [ ] **Step 3: Verify the sweep is complete and correctly scoped**

```bash
echo "--- should be empty ---"
git ls-files -z | xargs -0 grep -nE 'honest.scholar|honest_scholar|HonestScholar|HONEST_SCHOLAR' \
  | grep -v '2026-08-26-defendable-science-migration'

echo "--- the two migration documents must STILL contain the old name ---"
grep -c 'honest-scholar' docs/superpowers/specs/2026-08-26-defendable-science-migration-design.md
grep -c 'honest-scholar' docs/superpowers/plans/2026-08-26-defendable-science-migration.md

echo "--- no double-rewrite of the docs domain ---"
git ls-files -z | xargs -0 grep -n 'defendable-science\.science' || echo "none (correct)"
```

Expected: the first command prints nothing; the two counts are non-zero; the last prints `none (correct)`.

- [ ] **Step 4: Check the inbound links to the renamed spec resolve**

```bash
grep -rn '2026-07-21-defendable-science-integration-design' CLAUDE.md .defendable-science/config.yml
test -f docs/superpowers/specs/2026-07-21-defendable-science-integration-design.md && echo "target exists"
```

Expected: both files reference the new filename, and the target exists. The sweep rewrites the link text automatically because the old filename contains `honest-scholar`; this step confirms it rather than trusting it.

- [ ] **Step 5: Confirm the trailer convention was rewritten**

```bash
grep -rn 'DefendableScience-Skill' docs/superpowers/specs/2026-07-21-defendable-science-integration-design.md
```

Expected: one hit at the line that previously read `HonestScholar-Skill: research-init`.

- [ ] **Step 6: Run the hooks over everything touched**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass. `codespell` and `detect-secrets` are the ones that could object to a mass edit; if `detect-secrets` flags a renamed path, report it rather than regenerating the baseline unprompted.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit --no-gpg-sign -m "$(cat <<'MSG'
docs: rename honest-scholar to defendable-science throughout

Mechanical sweep of every remaining occurrence, including provenance
lines in research records and the dated integration spec, which is
renamed (keeping its date) with its two inbound links updated.

The two documents describing this migration are deliberately excluded:
they carry the former name in their "before" columns, and sweeping them
would destroy their meaning.

The former name now survives only in the seven commits carrying the old
HonestScholar-Skill: trailer, which are immutable and correctly so.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: Record the reversed dependency decision in `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (the research-workflow section, around lines 47–58 before Task 2's sweep)

**Interfaces:**
- Consumes: Task 2's sweep, which has already replaced the names in this section.
- Produces: nothing.

This is the one edit the sweep cannot do, because the *claim* is wrong rather than the name. After Task 2 the line reads "CLI via `uv tool install defendable-science` — deliberately **not** in `pyproject.toml`" — mechanically correct and factually false: the CLI **is** pinned in the `dev` group (`pyproject.toml`, `defendable-science==0.2.1`).

- [ ] **Step 1: Read the section as the sweep left it**

```bash
sed -n '45,60p' CLAUDE.md
```

- [ ] **Step 2: Replace the parenthetical about installation**

Find, in the paragraph beginning "The scientific work runs on":

```
(enabled in `.claude/settings.json`; CLI via `uv tool install defendable-science` — deliberately **not** in `pyproject.toml`)
```

Replace with:

```
(enabled in `.claude/settings.json`, marketplace pinned to `v0.2.1`; CLI pinned as `defendable-science==0.2.1` in the `dev` dependency group, so `uv sync` installs it in every devcontainer flavor and the version is reproducible from the lockfile)
```

- [ ] **Step 3: Update the keys sentence to the current CLI name**

In the "Research contact email" paragraph, the sweep will have produced `defendable-science keys set <NAME>` and `defendable-science keys check`. Confirm both read correctly, and add the short alias, so the sentence names the command a reader will actually type:

```
set keys via `defendable-science keys set <NAME>` (hidden prompt, or `dsci keys set <NAME>`), and `defendable-science keys check` to verify presence
```

- [ ] **Step 4: Verify the claim the section now makes**

```bash
grep -n 'defendable-science==0.2.1' pyproject.toml
grep -n 'uv tool install' CLAUDE.md || echo "no stale uv tool install claim (correct)"
uv run --no-sync defendable-science --version
uv run --no-sync dsci --version
```

Expected: the pin exists in `pyproject.toml`; no `uv tool install` claim remains; both CLI names report the same version, confirming the alias documented in Step 3 is real.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit --no-gpg-sign -m "$(cat <<'MSG'
docs(claude): correct the defendable-science install rationale

The sweep renamed this sentence but could not fix it: it claimed the CLI
is installed with `uv tool install` and "deliberately not in
pyproject.toml". It is now pinned as defendable-science==0.2.1 in the dev
group, which is a default group -- so uv sync installs it in every
devcontainer flavor and the version is reproducible from the lockfile.

Also documents the marketplace pin and the `dsci` alias, both verified.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: Whole-migration verification and PR

**Files:** none.

- [ ] **Step 1: Run the spec's verification checklist end to end**

```bash
echo "--- 1. no stale name outside the migration docs ---"
git ls-files -z | xargs -0 grep -nE 'honest.scholar|honest_scholar|HonestScholar|HONEST_SCHOLAR' \
  | grep -v '2026-08-26-defendable-science-migration' || echo "clean"

echo "--- 2. directory state ---"
test -d .defendable-science && echo ".defendable-science exists"
test -d .honest-scholar && echo "FAIL: .honest-scholar still exists" || echo ".honest-scholar gone"

echo "--- 3. the decisive check ---"
uv run --no-sync python -c "from defendable_science.core.config import load_config; c = load_config(); print(sorted(c)); assert c, 'config not found at the default path'"

echo "--- 4. CLI ---"
uv run --no-sync defendable-science doctor

echo "--- 5. settings ---"
python3 -c "import json; json.load(open('.claude/settings.json')); print('valid json')"

echo "--- 6. hooks ---"
uv run pre-commit run --all-files

echo "--- 7. docs ---"
./tools/build-docs.sh
```

Expected: clean grep; `.defendable-science` present and `.honest-scholar` gone; the four config keys; `doctor` reporting the environment; valid JSON; all hooks passing; docs building.

- [ ] **Step 2: Open the PR**

Follow the **create-pr** skill (`.claude/skills/create-pr/SKILL.md`). The body must state plainly that plugin load is **not** verified by CI or by any local check — it is observable only after a Claude Code session restart — and must not claim otherwise.

- [ ] **Step 3: Report what remains manual**

After CI passes, tell the maintainer explicitly:
- restart the Claude session and confirm the `defendable-science` skills are listed;
- if any machine has `uv tool install honest-scholar`, run `uv tool uninstall honest-scholar` (none is installed in this container);
- if any machine holds `.honest-scholar/keys.json`, move it by hand to the `defendable-science` 0.2.1 default store — `$XDG_CONFIG_HOME/defendable-science/keys.json` (falling back to `~/.config/...`, mode `0600`), not `.defendable-science/keys.json` (that in-repo path is legacy/opt-in only, via `DEFENDABLE_SCIENCE_KEYS_PATH`, which this repo does not set) — it is gitignored, so no commit carries it and a fresh clone will not reveal its absence until a key lookup fails.

---

## Self-review

**Spec coverage.** Upstream change table → Task 1 (plugin, config dir) and Task 2 (names, trailer, docs domain); the env-var row needs no work and the spec says so. Decision 1 "rewrite every occurrence" → Task 2, including the dated spec rename. Decision 2 "pin v0.2.1" → Task 1 Steps 4–5, with the tag's existence verified upstream. Decision 3 "record the reversed dependency decision" → Task 3. Risk "a partial rename looks like success" → Task 1 Steps 1 and 3 bracket the change with the same check, before and after. Risk "secrets do not move with git" → Task 1 Step 7 proves the ignore still bites, and Task 4 Step 3 tells the maintainer to move any real key store. Risk "the plugin cannot be verified in-session" → Task 4 Steps 2 and 3 require saying so rather than claiming success. Every spec verification bullet maps to a Task 4 Step 1 command.

**Placeholders.** None: every step has the exact text or command and its expected output. The one variable is the `Refs #` line in Task 1's commit message, which carries an explicit instruction to delete it if no issue exists.

**Name consistency.** `.defendable-science/config.yml` is the path in Tasks 1, 2 and 4. `load_config()` is invoked identically in Task 1 Steps 1 and 3 and Task 4 Step 1, so the before/after comparison is like-for-like. The marketplace key, the plugin id `defendable-science@defendable-science` and the ref `v0.2.1` match between Task 1 Step 4 and its verification in Step 5. The two excluded filenames are written identically in the Global Constraints, the File Structure table, and Task 2 Steps 2–3.
