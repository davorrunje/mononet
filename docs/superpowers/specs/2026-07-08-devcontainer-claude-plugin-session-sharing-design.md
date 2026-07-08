# Devcontainer ↔ host: Claude plugin & session sharing

**Date:** 2026-07-08
**Status:** Draft (awaiting author review)
**Scope:** `.devcontainer/**` (all five flavors) + `.devcontainer/shared/**`. No change to the `mononet` package.

## 1. Problem

The devcontainer bind-mounts the **entire host `~/.claude`** into the container:

```jsonc
// .devcontainer/<flavor>/devcontainer.json
"containerEnv": { "CLAUDE_CONFIG_DIR": "/home/vscode/.claude" },
"mounts": [ "source=${localEnv:HOME}/.claude,target=/home/vscode/.claude,type=bind,consistency=cached" ]
```

Because the same user runs Claude Code **both on the host and in the devcontainer**, this corrupts shared state:

- **Plugins don't load.** Plugin config (`installed_plugins.json`, `known_marketplaces.json`) stores **`$HOME`-absolute paths**. The host writes `/home/<hostuser>/.claude/plugins/...`; in the container `$HOME` is `/home/vscode`, so the harness rejects them ("corrupted installLocation … expected a path inside /home/vscode/.claude"). A single file cannot be valid at both absolute paths.
- **Ownership conflicts.** Host runs (as root / a different uid) left some `~/.claude/plugins/marketplaces/*` dirs **root-owned**; the container's `vscode` user gets `EACCES` trying to repair them.

Net: `superpowers` (and its `brainstorming`/`writing-plans` skills) is installed+enabled but not usable in the container.

## 2. Goal & constraints

- **Share the *environment* across host and container** — specifically **sessions** (conversation transcripts) and **plugins**. Login/auth is *not* important (re-authenticating in the container is acceptable).
- **Portable** — must work for any contributor and in CI, with no hardcoded host username or host path, and must **never mutate the host's `~/.claude`** from inside the container.
- **Per-flavor** — the fix applies to all five flavors (`default`, `gpu-torch`, `gpu-jax`, `gpu-keras`, `proofs`) via the shared scripts, not copy-pasted logic.

## 3. Core constraint (why "just share the folder" fails)

Claude Code keys `~/.claude` contents by **absolute paths**, on two axes that both differ between host and container:

| state | keyed by | host | container |
|---|---|---|---|
| plugins | `$HOME`-absolute `installLocation` | `/home/<hostuser>/.claude/...` | `/home/vscode/.claude/...` |
| sessions | slug of the working directory | `-<hostuser-repo-path>` | `-workspaces-mononet` |

Sessions live in `~/.claude/projects/<slug(cwd)>/`, where `slug` is the cwd with `/` → `-`. The container's cwd is fixed (`/workspaces/mononet` → `-workspaces-mononet`); the host's is wherever the repo is checked out. So a blanket share of `~/.claude` gives you *both* trees but they never line up, and the plugin paths are invalid in one of the two environments.

**Therefore:** plugins must be **per-environment** (each environment owns its plugin install), and sessions must be shared by a **targeted, slug-mapping mount** — not a blanket folder share.

## 4. Design

Two independent halves.

### 4a. Container-local `~/.claude` + plugins provisioned at build

- **Drop** the whole-`~/.claude` bind mount. Give the container **its own `~/.claude`** as a **named Docker volume** (container-owned, persists across rebuilds so plugins aren't re-cloned every time). Defined once per flavor's `docker-compose.yml`.
- **Provision plugins declaratively** in `postCreateCommand` (`shared/post-create.sh`), via a new idempotent `shared/provision-claude-plugins.sh` driven by a checked-in manifest `.devcontainer/claude-plugins.txt`:

  ```
  # marketplace-source                         plugin@marketplace
  https://github.com/obra/superpowers.git      superpowers@superpowers-dev
  ```

  For each line: `claude plugin marketplace add <source>` (skip if present) then `claude plugin install <plugin> --scope user` (skip if already installed via `claude plugin list`). Writes only into the container-local volume, so the host is never touched. `plugin install`/`marketplace add` are local operations and **do not require login**.

- **Result:** the container has a correct, reproducible, container-owned plugin set; `superpowers` skills work; the host's plugin config is untouched.

### 4b. Sessions shared via a targeted slug-mapping mount

We share **only this repo's session directory**, mapping the host slug onto the container slug:

- `initializeCommand` (`shared/host-init.sh`, runs **on the host**, cwd = host repo path) computes the host session dir for this repo — `${HOME}/.claude/projects/<slug($PWD)>` — creates it if absent, and exposes it at a **stable, host-user-agnostic path** by symlinking it into the existing secrets dir: `${HOME}/.config/mononet-devcontainer/claude-session → …/projects/<slug>`. (This reuses the exact pattern already used to forward the host `gh` token.)
- `devcontainer.json` bind-mounts that stable path onto the container's fixed session dir:

  ```jsonc
  "source=${localEnv:HOME}/.config/mononet-devcontainer/claude-session,target=/home/vscode/.claude/projects/-workspaces-mononet,type=bind"
  ```

  (Docker resolves the host symlink to the real project dir.) This bind sits *over* the named-volume `~/.claude`, so `projects/-workspaces-mononet` is the shared host dir while everything else in `~/.claude` stays container-local.

- **Result:** host and container read/write the **same transcript files** for this project → sessions continue across environments, while plugins/settings remain per-environment.

## 5. Concrete changes

- `.devcontainer/claude-plugins.txt` — **new**, declarative plugin manifest (superpowers to start).
- `.devcontainer/shared/provision-claude-plugins.sh` — **new**, idempotent marketplace-add + plugin-install loop over the manifest; non-fatal on failure (network/CI).
- `.devcontainer/shared/post-create.sh` — call `provision-claude-plugins.sh`.
- `.devcontainer/shared/host-init.sh` — additionally compute the host session slug and create the `claude-session` symlink in the secrets dir (non-fatal on failure, like the gh-token block).
- `.devcontainer/<flavor>/devcontainer.json` (×5) — replace the whole-`~/.claude` bind with (a) a named-volume `~/.claude` and (b) the targeted session bind mount.
- `.devcontainer/<flavor>/docker-compose.yml` (×5) — declare the named volume.

## 6. Alternatives considered

- **Keep sharing the whole `~/.claude`** — rejected: the absolute-path conflict is unfixable without hacks that break one side.
- **Neutral shared paths** (identical `CLAUDE_CONFIG_DIR` *and* repo checkout path on host and container, e.g. `/claude-shared` + a fixed repo path) — would let plugins *and* sessions share verbatim, but requires configuring the **host** (outside the repo) and diverges from devcontainer conventions (`/workspaces`, `~/.claude`). Rejected for portability; noted as an optional power-user setup.
- **Container mirrors host paths** (container home = `/home/<hostuser>`, workspace at the host path) — seamless sharing but **hardcodes one contributor's layout**; breaks others + CI. Rejected.
- **Repair-in-place** (post-create rewrites `installLocation`/chowns the shared dir) — mutates the host's config, breaking the host view. Rejected.

## 7. Risks & open questions

1. **Slug-algorithm fidelity.** `host-init.sh` recomputes the session slug; it must match Claude Code's exact rule (observed: `/`→`-`, leading `-`; behaviour for `.`/other chars unverified). *Mitigation:* verify against a real host session dir during implementation; if fragile, discover the dir by listing `~/.claude/projects` and matching rather than recomputing.
2. **Symlink resolution through the bind mount.** Relies on Docker resolving a host-side symlink source. To verify on the target Docker/OS; fall back to writing the resolved real path into a secrets file that a wrapper mount consumes.
3. **CI / no host `~/.claude`.** In CI there is no host session dir and possibly no network for `marketplace add`. Both new scripts must be **non-fatal** (skip cleanly) so container build never breaks.
4. **Named-volume staleness.** A persisted `~/.claude` volume could accumulate stale plugin versions; `provision-claude-plugins.sh` should `plugin update` or be resettable (documented `docker volume rm`).
5. **Concurrent host+container writes to the same session** could interleave transcript writes if both run at once on the same project. Acceptable (users won't typically do both simultaneously); documented.

## 8. Out of scope / future

- Sharing settings, memory, or auth (auth is explicitly disposable here).
- The "neutral shared paths" power-user mode (§6) — could be a documented opt-in later.
- Any `mononet` package change.

## 9. Testing / rollout

- Rebuild one flavor (`gpu-torch`), confirm: `claude plugin list` shows `superpowers` (skills available); a session started on the host is resumable in the container and vice-versa; the host's `~/.claude` is byte-unchanged after a container run.
- Confirm CPU `default` + a GPU flavor + `proofs` build cleanly with no host `~/.claude` present (CI-like).
- Then roll the shared-script + per-flavor mount changes across all five.
