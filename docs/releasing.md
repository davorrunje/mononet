# Releasing

Maintainer runbook for cutting a `mononet` release to PyPI. Publishing uses
**PyPI Trusted Publishing** (OIDC) — no API tokens or stored secrets.

## One-time setup (maintainer, web UI)

Do this once before the first release.

1. **GitHub Environments.** In the repository settings, create two
   environments: `pypi` and `testpypi`.
2. **PyPI pending publisher** — at <https://pypi.org/manage/account/publishing/>,
   add a pending publisher:
   - PyPI Project Name: `mononet`
   - Owner: `davorrunje`
   - Repository name: `mononet`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. **TestPyPI pending publisher** — repeat at
   <https://test.pypi.org/manage/account/publishing/> with Environment name
   `testpypi`.

If any of these values is wrong, the first upload fails at the OIDC exchange
with a "not a trusted publisher" error.

## Version bumping

The version lives in `pyproject.toml` and is the single source of truth. Use
the **Bump Version** GitHub Action (Actions tab → Bump Version → Run workflow),
which runs `uv version --bump <type>` and opens a PR. Options:

| Bump | From `0.0.0a0` | Notes |
|------|----------------|-------|
| `alpha` | `0.0.0a1` | iterate the alpha |
| `beta` | `0.0.0b1` | |
| `rc` | `0.0.0rc1` | |
| `stable` | `0.0.0` | promote prerelease to final |
| `patch` | `0.0.1` | |
| `minor` | `0.1.0` | |
| `major` | `1.0.0` | |

`alpha`/`beta`/`rc`/`stable` only work standalone when the current version is
already a pre-release. From a stable version, bump a release component first
(`patch`/`minor`/`major`); a bare `alpha` bump from a stable version errors.

## Per-release flow

1. *(If benchmarks changed)* in a `gpu-*` devcontainer, run
   `./tools/execute-benchmarks.sh`, sanity-check `git diff docs/benchmarks/`,
   and commit the regenerated notebooks.
2. Run **Bump Version** with the desired bump type; merge the resulting PR into
   `main`.
3. *(Optional rehearsal)* Actions tab → **Publish** → Run workflow →
   `target: testpypi` (which is the default). Verify the project page on
   <https://test.pypi.org/project/mononet/> and a clean install:
   `pip install -i https://test.pypi.org/simple/ mononet==<version>`.
4. Create a **GitHub Release**: tag `v<version>` (e.g. `v0.0.0a0`), target the
   merge commit on `main`, mark it **pre-release** for alphas/betas/rcs, and
   write release notes.
5. Publishing the release fires the **Publish** workflow to real PyPI. It fails
   fast if the tag does not match the `pyproject.toml` version. The `v*.*.*`
   tag created by the release triggers the **Docs** workflow to deploy the
   versioned docs.

## Notes

- Publishing does not re-run the test matrix; releases are expected to be cut
  from a `main` whose `build.yml` checks are green.
- A real-PyPI upload cannot be replaced, only yanked. Use the TestPyPI
  rehearsal for first-of-its-kind releases.
