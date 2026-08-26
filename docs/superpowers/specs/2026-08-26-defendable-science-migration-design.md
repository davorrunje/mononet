# honest-scholar → defendable-science migration — design

**Date:** 2026-08-26
**Status:** approved (brainstorming)
**Owner:** Davor Runje
**Upstream:** [defendable.science](https://defendable.science/) ·
[`davorrunje/defendable-science`](https://github.com/davorrunje/defendable-science)

## Purpose

The research-workflow plugin this repository depends on was renamed from
**honest-scholar** to **defendable-science** in its `0.2.0` release. Upstream
ADR-0035 states the reason: "the name now describes what the tool verifies —
that the work can be defended — rather than a virtue it cannot audit." The
rename is breaking in every direction and ships **no deprecation shim**; the
`honest-scholar` PyPI distribution is abandoned at `0.1.1` with no forwarding
release.

This document specifies the migration of `mononet` onto the new plugin, CLI and
config layout.

## What upstream changed

Taken from the project's `CHANGELOG.md` for `0.2.0`, which is the authoritative
list:

| | Before | After |
|---|---|---|
| Plugin install | `honest-scholar@honest-scholar` | `defendable-science@defendable-science` |
| CLI | `honest-scholar` (alias `hsch`) | `defendable-science` (alias `dsci`) |
| PyPI distribution | `honest-scholar` | `defendable-science` |
| Python module | `honest_scholar` | `defendable_science` |
| Project config dir | `.honest-scholar/` | `.defendable-science/` |
| Env vars | `HONEST_SCHOLAR_KEYS_PATH`, `HONEST_SCHOLAR_LIVE` | `DEFENDABLE_SCIENCE_KEYS_PATH`, `DEFENDABLE_SCIENCE_LIVE` |
| Commit trailer | `HonestScholar-Skill:` | `DefendableScience-Skill:` |
| Docs domain | `honest-scholar.science` | `defendable.science` |
| Repository | `davorrunje/honest-scholar` | `davorrunje/defendable-science` |

Verified against the installed package rather than taken on trust:
`defendable_science` 0.2.1 exposes console scripts `defendable-science` and
`dsci`, and `core/config.py` declares
`DEFAULT_CONFIG_PATH = Path(".defendable-science/config.yml")`. The package
contains **no** remaining reference to the former name, confirming there is no
compatibility fallback.

Neither env var is referenced anywhere in this repository, so that row needs no
work here.

## Current state in this repository

29 files mention the former name, in five distinct roles:

| Area | Files | Role |
|---|---|---|
| `.claude/settings.json` | 1 | live plugin wiring (`enabledPlugins`, `extraKnownMarketplaces`) |
| `.honest-scholar/` | 2 | live config (`config.yml`, `rclone.conf.example`) |
| `.gitignore` | 1 | 6 lines covering `rclone.conf`, `keys.json`, `cache/` |
| `CLAUDE.md`, `datasets.yml`, `.claude/skills/create-pr/STYLE.md` | 3 | operative prose and examples |
| `docs/research/**`, `docs/superpowers/specs/2026-07-21-…` | 22 | research records and the integration spec |

The CLI is already declared: `defendable-science==0.2.1` sits in the `dev`
dependency group (`pyproject.toml`), added ahead of this work. No `keys.json`
exists in the working container.

## Decisions

### 1. Rewrite every occurrence

All 29 files are rewritten, including provenance lines such as "Drafted by the
honest-scholar `dataset` skill" and the dated integration spec.

The alternative — fixing only references that would dangle, and leaving
attributions as historical fact — has a real precedent: upstream's own changelog
leaves pre-`0.2.0` entries unedited because "those releases really did ship under
that name." It was considered and rejected in favour of a repository where the
former name survives in exactly one place: the existing commits carrying the
`HonestScholar-Skill:` trailer, which are immutable and correctly so — two
reachable from `origin/main` (eight across all refs, including unmerged
branches).

The dated spec `2026-07-21-honest-scholar-integration-design.md` is renamed to
`2026-07-21-defendable-science-integration-design.md`, keeping its date. Its two
inbound links (`CLAUDE.md`, the config header) are updated.

### 2. Pin the marketplace to `v0.2.1`

`.claude/settings.json` pins the marketplace ref to `v0.2.1`, matching the
`defendable-science==0.2.1` CLI pin so plugin and CLI move in lockstep. This
continues the existing discipline — the former plugin was pinned to `v0.1.0` —
and is deliberately *not* the `v0.2.0` shown in the upstream install snippet,
which is older than the current release. Plugin and CLI are versioned
independently upstream (ADR-0026), so the two pins are bumped together by hand.

### 3. Record the reversed dependency decision

`CLAUDE.md` currently states the CLI is installed via `uv tool install
honest-scholar` and is "deliberately **not** in `pyproject.toml`". That decision
is now reversed: the CLI is a pinned entry in the `dev` group. The replacement
rationale — reproducibility from the lockfile, and every devcontainer flavor
getting it from `uv sync` without a manual step — is recorded in `CLAUDE.md`
rather than left as an unexplained contradiction. `dev` is a default group, so
the CLI is also installed in CI; that cost is already being paid.

## Risks

**A partial rename looks like success.** `load_config` returns `{}` for a missing
file so that "callers can treat an unconfigured project as all defaults". If the
directory rename is missed or half-applied, the CLI does not error — it runs with
defaults, and every binding in `config.yml` (engineering backend, experiment
backend, literature anchors) is silently ignored. Verification must therefore
demonstrate that configuration is *read from the new path*, not merely that
commands exit 0.

**Secrets do not move with git.** `.honest-scholar/keys.json` is gitignored. It
does not exist in the working container, so nothing is at risk here, but on
`defendable-science` 0.2.1 the default key store is out of repo —
`$XDG_CONFIG_HOME/defendable-science/keys.json` (falling back to
`~/.config/...`, mode `0600`) — so any machine holding the old file should move
it there, not into `.defendable-science/keys.json`. The in-repo path is a
legacy/opt-in location, reachable only by setting `DEFENDABLE_SCIENCE_KEYS_PATH`,
which this repo does not set. `git mv` will not carry either path and a fresh
clone will not reveal its absence until a key lookup fails.

**The plugin cannot be verified in-session.** Whether Claude Code resolves and
loads `defendable-science@defendable-science` is only observable after the
session restarts. The migration can verify that `settings.json` is well-formed
and that the referenced tag exists upstream; actual plugin load is a manual
check, and must be reported as such rather than assumed.

## Verification

- [ ] `grep -rn "honest.scholar\|honest_scholar"` over the worktree returns
      nothing outside `.git/`
- [ ] `.defendable-science/config.yml` exists and `.honest-scholar/` does not
- [ ] the CLI reads configuration from the new path. The decisive check, because
      it fails loudly on a half-done rename rather than exiting 0:

      ```bash
      uv run python -c "from defendable_science.core.config import load_config; \
        c = load_config(); print(sorted(c)); assert c, 'config not found at the default path'"
      ```

      It must print the four top-level keys
      (`engineering_backend`, `experiment_backend`, `literature`, `mirror`).
      Confirmed to discriminate: run against the pre-migration tree it prints
      `[]` and the assertion fails, because `load_config` silently defaults.
- [ ] `uv run defendable-science doctor` runs, and `dsci` resolves to the same
      entry point
- [ ] `.claude/settings.json` is valid JSON and names marketplace ref `v0.2.1`;
      the tag exists in `davorrunje/defendable-science`
- [ ] `uv run pre-commit run --all-files` passes, including `detect-secrets`
      (`.secrets.baseline` contains no path under the old directory, so no
      baseline regeneration is expected — confirm rather than assume)
- [ ] `./tools/build-docs.sh` succeeds; `docs/research/` is excluded from the
      Sphinx build but `CLAUDE.md` and the specs are not
- [ ] **manual, after session restart:** the plugin loads and its skills are
      listed

## Follow-ups

- The `docs/research/dashboard.md` header is machine-generated
  (`generated by honest-scholar:progress — do not edit`). Rewriting it by hand is
  correct for consistency now, but the next `progress` run regenerates it; no
  action needed beyond expecting that.
- If any machine still has `uv tool install honest-scholar`, run
  `uv tool uninstall honest-scholar`. None is installed in this container.
