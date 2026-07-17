# Dataset Tooling & Mirror Architecture — digest

**Date:** 2026-07-17 · **For:** `scholar` `dataset` capability (sub-spec 3). · **Status:** verified-source digest; migrates to the plugin's `resources/references/`.

## Bottom line
Thin custom **`datasets.yml`** + **rclone** (private mirror) + **pooch** (Tier-B fetch) + a ~200-line resolution-chain core. Reject DVC/DataLad/lakeFS/Quilt on dependency-weight grounds. Adopt `huggingface_hub` opportunistically for HF-hosted/gated data. git-annex is the fallback single-tool option if a Haskell binary is acceptable.

## 1. Tooling review

| Tool | Model | Remotes | Dep weight | Verdict |
|---|---|---|---|---|
| **Git LFS** | pointer + sharded store; clean/smudge | HTTPS LFS server; no native S3/rclone | single Go binary | **Tier A only** (small, redistributable, in-repo); server-centric remote wrong for mirror |
| **git-annex** | key store + committed symlinks; location branch | **native rclone + S3 special remotes** | Haskell binary | closest single-tool match, but adopting its worldview is overkill vs calling rclone ourselves |
| **DataLad** | git-annex + provenance (`run`/`rerun`) | git-annex siblings | **heaviest** (Python + git + git-annex) | best provenance, but triple dep conflicts light-dep posture. No |
| **DVC** | `.dvc`/`dvc.yaml` + MD5 cache | cache↔remote + re-verify; GDrive/S3/…; **no auto-fallback** | **~40 direct deps** | does cache→remote+checksums OOTB, but dep mass disqualifies; fights no-Rust ethos |
| **HF Hub** (`huggingface_hub`) | git+LFS/Xet; content-addressed cache | Hub-hosted; `revision=`; `hf cache verify` | light (~9 py + optional Rust `hf-xet`) | strong **for HF-hosted/gated**; couples to HF as host — use opportunistically |
| **`datasets`** | Arrow on hub | via hub | heavy (pyarrow/pandas/…) | **don't depend on it in the plugin** (loading lib, not a fetcher) |
| **rclone** | transfer engine | **70+ backends** (GDrive/S3/B2/R2/SFTP/WebDAV…); `remote:path`; verifies on transfer | **single static Go binary** | **adopt as the mirror engine** |
| **pooch** | name-addressed cache; `registry.txt`; re-verify | HTTP/FTP/SFTP + `doi:` (Zenodo/Figshare); custom downloaders | **3 pure-Python deps** | **adopt as the Tier-B fetcher** |
| **lakeFS** | git-like over object store | own S3-compatible endpoint | **always-on server + DB** | no — far too heavy |
| **Quilt** | versioned S3 packages | S3 only | boto3 | niche S3-only fallback |

## 2. Private-mirror architecture — rclone

Single dep-free Go binary; Google Drive + S3/B2/R2/MinIO/SFTP/WebDAV/~65 more; `remote:path`; verifies integrity on transfer; `rclone check`/`hashsum` on demand.

Registry names a remote + path; the credential-bearing config lives **outside the repo**:
```yaml
# datasets.yml (committed, public)
mirror:
  rclone_remote: <name>          # logical name only, no secrets
  base_path: "<project>/datasets"
  hash: md5                       # lowest common denominator GDrive/S3
```
Keep private while registry stays public: (1) gitignore `rclone.conf`, point via `RCLONE_CONFIG=$PWD/.rclone/rclone.conf`, commit `rclone.conf.example` (name/type only); or (2) **env-var remotes** (`RCLONE_CONFIG_<NAME>_TYPE=drive`, `_TOKEN=…`) from CI secrets — preferred for CI. Caveats: `rclone obscure` is **not** encryption (never commit it); env-var remote names use underscores; GDrive service account for CI, MD5 present on all files, 750 GiB/day upload quota.

Mirror × tier: A → redundant (skip, or large redistributable blobs); B → **primary durability** vs link-rot (populate on first fetch); C → **essential** (only re-fetchable copy after one-time gated acquisition).

## 3. Resolution chain `fetch(name)` — verify fixity at every hop
```
entry = registry[name]; target = cache_path(entry)
1 LOCAL CACHE:   if exists(target) and hash==expected: return; else discard-if-corrupt
2 PRIVATE MIRROR (if configured): rclone copyto mirror:base/<key> target; verify → return; else fall through
3 PUBLIC SOURCE  (Tier A git/LFS, B pooch/http/doi): fetch; assert hash==expected (hard fail); populate mirror; return
4 TIER C:        print instructions; wait for operator drop; assert hash; move to target; populate mirror; return
```
Standardize one hash across hops (**MD5** = GDrive+S3 common; use `rclone check --download` if backends' hash sets are disjoint). Expected hash lives **in the manifest** (single source of truth), never trust a backend-reported hash alone. Cache: gitignored; **content-addressed on the wire** (`<alg>/<hash>` → dedup, integrity==identity, like HF `blobs/`), name-addressed for use (symlink/copy).

## 4. Registry format
**Own a thin `datasets.yml`** (YAML: human-editable, comments, `pyyaml`). Do **not** adopt DVC's `.dvc`/`dvc.yaml` (bound to DVC's cache + ~40 deps; lacks multi-source fallback). Keep pooch's `registry.txt` (`name hash`) as an optional export. `hash` carries an `alg:` prefix (self-describing). Fixity philosophy from BagIt: an entry is valid only when every listed checksum verifies.

## 5. Plugin vs repo split
**Plugin:** manifest schema + loader/validator; resolution-chain + per-hop fixity; rclone integration (subprocess `copyto`/`check`/`hashsum`; config via `--config`/env); Tier-B fetchers (pooch or minimal http/doi core) + Tier-A git/LFS + Tier-C drop-and-verify; cache policy; `rclone.conf.example` generator + secret-hygiene docs. Dep contract: **rclone (binary subprocess, not a Python dep) + pyyaml + pooch**.
**Consumer:** the `datasets.yml` entries + checksums; the rclone remote config (real creds, untracked/CI); Tier-A blobs; the gitignored cache; any heavy loaders (`datasets`, pandas).

## 6. One skill or a set? — **one skill, verbs** `register | fetch | verify | mirror | audit` (+ `init`). Verbs share the schema/tier/cache/chain mental model and are sequential lifecycle steps; `fetch` = verify + mirror-populate composed; `audit` = verify across the manifest.

## Recommended stack
`datasets.yml` + **pooch** (Tier-B) + **rclone** (mirror, any backend) + ~200-line chain core; `huggingface_hub` opportunistic; reject DVC/DataLad/lakeFS/Quilt.

## Sources
Git LFS spec/api; git-annex internals/special_remotes; DataLad (JOSS 10.21105/joss.03262); DVC docs + pyproject; HF Hub docs + setup.py; rclone docs (/s3,/b2,/drive,/docs env-vars); pooch (fatiando.org/pooch) + pyproject; BagIt RFC 8493; Zenodo policies. (URLs verified this session.)

## Flags
No maintained rclone-native Git-LFS transfer agent found (LFS-over-rclone needs glue). DVC checksum-on-pull via maintainer forum post; ~40 deps = direct required; no-fallback inferred from docs. rclone env-var behavior partly via forum (standard). Dep lists from current `main` packaging (sets stable, versions drift).
