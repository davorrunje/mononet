# Mode Rename Migration (`absolute`→`mixed`, `switch`→`split`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the monotonic construction `mode` string values `absolute`→`mixed` and `switch`→`split` across the whole repo, as a hard break (no aliases), with **no behavior change** (equivalence vectors stay numerically identical).

**Architecture:** Pure identifier rename. The kernels/reference compare `mode` against string literals; every literal, default, type alias, test, committed artifact, and doc that carries the old value moves to the new value in lockstep. A helpful `ValueError` replaces silent breakage when the old strings are passed. This is phase 1 of the spec [`2026-07-13-monotone-constructions-init-and-ablation-design.md`](../specs/2026-07-13-monotone-constructions-init-and-ablation-design.md); it deliberately does **not** add the `alternate` mode (phase 2) or the composition-aware init.

**Tech Stack:** Python 3.11+, uv, pytest, ruff, mypy, pre-commit; PyTorch / JAX (Flax NNX) / Keras 3 backends selected via `MONONET_TEST_BACKEND`.

## Global Constraints

- Python 3.11+, line length 88 (ruff). Strict mypy. MyST field-list docstrings on public API.
- No Pydantic. Stdlib dataclasses only.
- Never commit to `main`; work on branch `spec/monotone-constructions-init-and-ablation` (already checked out).
- Commit with `git commit --no-gpg-sign`.
- `mode` valid values after this plan: **exactly** `"mixed"` and `"split"`. Default is `"mixed"`.
- **No behavior change:** the regenerated equivalence vectors must be numerically identical to the committed ones (only the `"mode"` string field and `REFERENCE_HASH` change).
- Backend tests skip uninstalled backends via `pytest.importorskip`; run each backend with `MONONET_TEST_BACKEND={torch|jax|keras}`.
- This environment is CPU-only; do not run GPU benchmarks. Migrating committed result JSONs is a data edit, not a re-run.

---

### Task 1: Core rename (config + reference + init) and hard-break error

**Files:**
- Modify: `mononet/core/config.py` (lines 16, 37, 49-50, 112, 123-124; docstrings 24-25, 97-98)
- Modify: `mononet/core/reference.py` (lines 106, 112, 120, 130; docstrings 99, 101, 148, 150)
- Modify: `mononet/core/init.py` (docstring line 6, 112 — keep the `absolute_init_params` symbol name; only prose)
- Check/Modify: `mononet/core/numerics.py` (grep for stray mode literals/comments)
- Test: `tests/core/test_config.py`, `tests/core/test_reference_dense.py`, `tests/core/test_reference_residual.py`

**Interfaces:**
- Produces: `Mode = Literal["mixed", "split"]` in `mononet/core/config.py`; default `mode="mixed"` on `MonoConfig`/`MonoResidualConfig`; `reference.monotonic_dense`/`monotonic_residual` accept `"mixed"`/`"split"`; `MonoConfig(mode="absolute"|"switch")` raises `ValueError` naming the replacement.

- [ ] **Step 1: Write the failing hard-break test** (new behavior)

Add to `tests/core/test_config.py`:

```python
import pytest
from mononet.core.config import MonoConfig, MonoResidualConfig
from mononet.core.types import ActivationSpec


@pytest.mark.parametrize("old,new", [("absolute", "mixed"), ("switch", "split")])
def test_old_mode_names_rejected_with_hint(old, new):
    with pytest.raises(ValueError, match=new):
        MonoConfig(units=4, mode=old)
    with pytest.raises(ValueError, match=new):
        MonoResidualConfig(units=4, mode=old, activation=ActivationSpec("relu"))


def test_from_dict_rejects_old_mode():
    with pytest.raises(ValueError, match="mixed"):
        MonoConfig.from_dict(
            {"units": 4, "mode": "absolute", "activation": {"name": "relu"},
             "convex_fraction": 0.5, "init": {"scheme": "he_normal", "seed": None},
             "bias": True}
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/core/test_config.py::test_old_mode_names_rejected_with_hint -v`
Expected: FAIL (currently `mode="absolute"` is accepted, no error raised).

- [ ] **Step 3: Rename the enum + add the migration error in `config.py`**

Replace line 16:
```python
Mode = Literal["split", "mixed"]
```
Change both defaults (lines 37, 112): `mode: Mode = "mixed"`.

Add a module-level map after line 16:
```python
_RENAMED_MODES = {"absolute": "mixed", "switch": "split"}
```

Replace **both** `__post_init__` validation blocks (config.py:49-50 and 123-124) with:
```python
        if self.mode in _RENAMED_MODES:
            raise ValueError(
                f"mode {self.mode!r} was renamed; use "
                f"{_RENAMED_MODES[self.mode]!r} instead"
            )
        if self.mode not in ("mixed", "split"):
            raise ValueError(f"mode must be 'mixed' or 'split'; got {self.mode!r}")
```
Update the `:param mode:` docstrings (24-25, 97-98) to describe `"mixed"` (default) / `"split"`.

- [ ] **Step 4: Rename the branches in `reference.py`**

Line 106 `if mode == "switch":` → `if mode == "split":`
Line 112 `if mode == "absolute":` → `if mode == "mixed":`
Line 120 error → `raise ValueError(f"mode must be 'mixed' or 'split'; got {mode!r}")`
Line 130 default → `mode: str = "mixed"`
Update docstrings 99, 101, 148, 150 (`"absolute"`/`"switch"` → `"mixed"`/`"split"`).

- [ ] **Step 5: Update `init.py` prose and check `numerics.py`**

In `init.py`, update the module docstring (line 6, `mode="absolute"` → `mode="mixed"`) and the `absolute_init_params` docstring (line 112, "the ``absolute`` construction" → "the ``mixed`` construction"). **Do not rename the function symbol.**

Run: `git grep -n '"absolute"\|"switch"' mononet/core/numerics.py` — if any mode literals appear, rename them; if only comments, update the comment text.

- [ ] **Step 6: Update core tests to new strings**

In `tests/core/test_config.py` update the existing fixtures/assertions: `mode="absolute"` → `mode="mixed"`, `mode="switch"` → `mode="split"`, and `cfg.mode == "absolute"` → `cfg.mode == "mixed"` (lines ~15, 26, 40, 48).
In `tests/core/test_reference_dense.py`: parametrize list (line 28) `["switch","absolute"]` → `["split","mixed"]`; positional call args (lines 24, 50, 51) `"switch"`→`"split"`, `"absolute"`→`"mixed"`.
In `tests/core/test_reference_residual.py`: `mode="switch"` → `mode="split"` (lines 24, 44, 60, 63).

- [ ] **Step 7: Run core tests green**

Run: `uv run pytest tests/core -v`
Expected: PASS (including the new hard-break tests).

- [ ] **Step 8: Verify no stray old mode literals in core**

Run: `git grep -nE '"(absolute|switch)"' mononet/core`
Expected: only the `_RENAMED_MODES` keys in `config.py`. Anything else must be fixed.

- [ ] **Step 9: Commit**

```bash
git add mononet/core tests/core
git commit --no-gpg-sign -m "refactor(core): rename mode absolute->mixed, switch->split; reject old names"
```

---

### Task 2: torch backend rename

**Files:**
- Modify: `mononet/torch/_kernels.py` (lines 92, 98, 105; docstrings 86, 88, 90)
- Modify: `mononet/torch/layers.py` (lines 82, 98, 164; docstrings 63, 136-137)
- Test: `tests/torch/*` (9 files — see below)

**Interfaces:**
- Consumes: `Mode` from `mononet.core.config` (Task 1).
- Produces: torch `MonoLinear`/`MonoResidual` default `mode="mixed"`; kernel accepts `"mixed"`/`"split"`.

- [ ] **Step 1: Rename kernel branches + error**

`mononet/torch/_kernels.py`: line 92 `== "switch"` → `== "split"`; line 98 `== "absolute"` → `== "mixed"`; line 105 error → `raise ValueError(f"mode must be 'mixed' or 'split'; got {mode!r}")`. Update docstrings 86/88/90.

- [ ] **Step 2: Rename layer defaults + init guard**

`mononet/torch/layers.py`: line 82 `mode: Mode = "mixed"`; line 98 `if mode == "mixed" and init is None:`; line 164 `mode: Mode = "mixed"`. Update docstrings 63, 136-137.

- [ ] **Step 3: Update torch tests to new strings**

Apply across `tests/torch/`:
- `test_default_mode.py:14,21` assertions `.mode == "absolute"` → `== "mixed"`.
- `test_default_activation.py:12` `mode="switch"` → `"split"`.
- `test_absolute_init.py:14,26,37` `mode="absolute"` → `"mixed"`; `:44` `mode="switch"` → `"split"`.
- `test_mono_residual_gate.py` (all `mode="absolute"` → `"mixed"`).
- `test_mono_residual_subdepth.py` (`mode="absolute"` → `"mixed"`; `:68` `"switch"` → `"split"`).
- `test_public_api.py:21,33` `mode="switch"` → `"split"`.
- `test_property_monotonic.py:23` `sampled_from(["switch","absolute"])` → `["split","mixed"]`; `:41` `mode="switch"` → `"split"`.
- `test_deep_residual.py:21,57` positional `"absolute"` → `"mixed"`.
- `test_deep_init.py:19,22,26` `mode="absolute"` → `"mixed"`.

- [ ] **Step 4: Run torch tests green**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch -v`
Expected: PASS.

- [ ] **Step 5: Verify no stray literals**

Run: `git grep -nE '"(absolute|switch)"' mononet/torch tests/torch`
Expected: none.

- [ ] **Step 6: Commit**

```bash
git add mononet/torch tests/torch
git commit --no-gpg-sign -m "refactor(torch): rename mode absolute->mixed, switch->split"
```

---

### Task 3: jax backend rename

**Files:**
- Modify: `mononet/jax/_kernels.py` (lines 81, 87, 93; docstrings 74, 77, 79)
- Modify: `mononet/jax/layers.py` (lines 90, 105, 182; docstrings 154, 168)
- Test: `tests/jax/*` (6 files)

**Interfaces:** Produces jax `MonoLinear`/`MonoResidual` default `mode="mixed"`; kernel accepts `"mixed"`/`"split"`.

- [ ] **Step 1: Rename kernel branches + error**

`mononet/jax/_kernels.py`: line 81 `== "switch"` → `== "split"`; line 87 `== "absolute"` → `== "mixed"`; line 93 error → `"mode must be 'mixed' or 'split'; got {mode!r}"`. Docstrings 74/77/79.

- [ ] **Step 2: Rename layer defaults + init guard**

`mononet/jax/layers.py`: line 90 `mode: Mode = "mixed"`; line 105 `if mode == "mixed" and init is None:`; line 182 `mode: Mode = "mixed"`. Docstrings 154, 168.

- [ ] **Step 3: Update jax tests**

`tests/jax/`: `test_absolute_init.py` (3), `test_default_activation.py` (1), `test_mono_residual_gate.py` (7), `test_mono_residual_subdepth.py` (10), `test_property_monotonic.py` (1 `sampled_from`), `test_public_api.py` (2) — apply the same `absolute→mixed`, `switch→split` substitutions as Task 2 (same fixture/assertion/parametrize shapes).

- [ ] **Step 4: Run jax tests green**

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/jax -v`
Expected: PASS.

- [ ] **Step 5: Verify + commit**

```bash
git grep -nE '"(absolute|switch)"' mononet/jax tests/jax   # expect none
git add mononet/jax tests/jax
git commit --no-gpg-sign -m "refactor(jax): rename mode absolute->mixed, switch->split"
```

---

### Task 4: keras backend rename

**Files:**
- Modify: `mononet/keras/_kernels.py` (lines 83, 89, 95; docstrings 74, 79, 81)
- Modify: `mononet/keras/layers.py` (lines 76, 93, 205)
- Test: `tests/keras/*` (7 files)

**Interfaces:** Produces keras `MonoDense`/`MonoResidual` default `mode="mixed"`; kernel accepts `"mixed"`/`"split"`.

- [ ] **Step 1: Rename kernel branches + error**

`mononet/keras/_kernels.py`: line 83 `== "switch"` → `== "split"`; line 89 `== "absolute"` → `== "mixed"`; line 95 error → `"mode must be 'mixed' or 'split'; got {mode!r}"`. Docstrings 74/79/81.

- [ ] **Step 2: Rename layer defaults + init guard**

`mononet/keras/layers.py`: line 76 `mode: Mode = "mixed"`; line 93 `self._absolute_default = mode == "mixed" and init is None` (keep the attribute name `_absolute_default`); line 205 `mode: Mode = "mixed"`.

- [ ] **Step 3: Update keras tests**

`tests/keras/`: `test_default_activation.py` (1), `test_absolute_init.py` (3), `test_mono_residual_subdepth.py` (~10), `test_mono_residual_gate.py` (7), `test_property_monotonic.py:26` (`sampled_from`), plus the two assertion sites: `test_public_api.py:30` `clone.mode == "absolute"` → `== "mixed"`, `test_coverage_gaps.py:44` `cfg["mode"] == "switch"` → `== "split"`. Apply `absolute→mixed`, `switch→split`.

- [ ] **Step 4: Run keras tests green**

Run: `MONONET_TEST_BACKEND=keras uv run pytest tests/keras -v`
Expected: PASS.

- [ ] **Step 5: Verify + commit**

```bash
git grep -nE '"(absolute|switch)"' mononet/keras tests/keras   # expect none
git add mononet/keras tests/keras
git commit --no-gpg-sign -m "refactor(keras): rename mode absolute->mixed, switch->split"
```

---

### Task 5: Regenerate equivalence vectors (values only) + REFERENCE_HASH

**Files:**
- Modify: `tools/regenerate-cases.py` (mode-value elements in grids at lines 62-70, 115-118, 152-159 — **not** the filename slugs)
- Regenerate: `tests/equivalence/cases/{mono_linear,mono_residual}/*.json` (the `"mode"` field), `tests/equivalence/cases/REFERENCE_HASH`
- Test: `tests/equivalence/*`

**Interfaces:** Consumes the renamed `reference.py` (Task 1). Produces vectors whose `params["mode"]` is `"mixed"`/`"split"`, numerically identical otherwise.

- [ ] **Step 1: Read the generator grids**

Run: `sed -n '58,160p' tools/regenerate-cases.py` — identify the mode-string element in each tuple (the standalone `"switch"`/`"absolute"`, distinct from the name-slug first element like `"4x2x3-switch-relu"`).

- [ ] **Step 2: Edit the mode-value elements only**

In each grid tuple change the standalone mode string: `"switch"` → `"split"`, `"absolute"` → `"mixed"`. **Leave the filename-slug first element unchanged** (keeps RNG seed → identical numbers).

- [ ] **Step 3: Verify no standalone old mode literal remains**

Run: `python3 - <<'PY'
import re, pathlib
src = pathlib.Path("tools/regenerate-cases.py").read_text()
# standalone mode strings sit alone in a tuple slot: preceded by ", " and followed by ","
bad = re.findall(r',\s*"(absolute|switch)"\s*,', src)
print("remaining standalone old-mode literals:", bad)
PY`
Expected: `[]` (any `absolute`/`switch` left are only inside longer slug strings).

- [ ] **Step 4: Regenerate**

Run: `uv run python tools/regenerate-cases.py`
This rewrites the case JSONs and `REFERENCE_HASH`.

- [ ] **Step 5: Confirm the diff is values-only (no numeric change)**

Run: `git diff --stat tests/equivalence/cases` then `git diff tests/equivalence/cases/mono_linear/4x2x3-switch-relu.json`
Expected: the only content change per case file is `"mode": "switch"` → `"mode": "split"` (and `"absolute"`→`"mixed"` in the `-abs-` files); `REFERENCE_HASH` changes. **No numeric fields change.** If any number changed, a filename slug was accidentally edited — revert and redo Step 2.

- [ ] **Step 6: REFERENCE_HASH guard + equivalence on all backends**

```bash
git add mononet tools tests/equivalence/cases
uv run pre-commit run reference-hash --all-files
MONONET_TEST_BACKEND=torch uv run pytest tests/equivalence
MONONET_TEST_BACKEND=jax   uv run pytest tests/equivalence
MONONET_TEST_BACKEND=keras uv run pytest tests/equivalence
```
Expected: hook PASS; all three backends PASS.

- [ ] **Step 7: Commit**

```bash
git add tools tests/equivalence
git commit --no-gpg-sign -m "test(equivalence): regenerate vectors with mixed/split mode names + REFERENCE_HASH"
```

---

### Task 6: benchmarks rename (`_common` + scripts + tests)

**Files:**
- Modify: `benchmarks/_common/search.py` (`_ALL_FLAVORS` lines 247-252; `flavor_name` docstring 29)
- Modify: `benchmarks/search.py` (`_parse_flavors` `valid_modes` line 54; docstring 48; `--flavors` help 70)
- Modify: `benchmarks/_common/search_spaces.py` (line 26 `Literal`; **line 70** `mode == "absolute"` → `== "mixed"`; docstring 41)
- Modify: `benchmarks/_common/config.py` (line 66 `Literal`), `benchmarks/_common/config_io.py` (line 19 `Literal`)
- Modify: `benchmarks/_common/make_tables.py` (lines 92-95 flavor lookups)
- Modify standalone scripts: `benchmarks/deep_residual_run.py:24`, `benchmarks/deep_init_run.py:31-33`, `benchmarks/run.py:64,123`, `benchmarks/monoresidual_gate_ablation.py:79,80,127,129`, `benchmarks/monoresidual_gate_trap.py:84,86`, `benchmarks/monoresidual_gate_scale.py:64,66`, `benchmarks/loan_size_ladder_run.py:92,150`
- Test: `tests/benchmarks/*`

**Interfaces:** Produces `_ALL_FLAVORS` with `("split"|"mixed", …)` triples; `flavor_name` emits `mixed-*`/`split-*`; `_parse_flavors` accepts `mixed`/`split` kinds; `search_spaces` gates `convex_fraction` on `mode == "mixed"`.

- [ ] **Step 1: Rename `_ALL_FLAVORS` and flavor plumbing**

`benchmarks/_common/search.py:247-252`: `"switch"` → `"split"`, `"absolute"` → `"mixed"` in all six triples.
`benchmarks/search.py:54`: `valid_modes = {"split", "mixed"}`; update docstring 48 and help 70 example strings to `split-plain,mixed-deep`.

- [ ] **Step 2: Rename type aliases + the behavioral branch**

Set `Literal["split", "mixed"]` in `config.py:66`, `config_io.py:19`, `search_spaces.py:26`, and `benchmarks/run.py:64` (and `:123` `choices=["split","mixed"]`).
`search_spaces.py:70`: `... if mode == "mixed" else 0.5`.

- [ ] **Step 3: Rename `make_tables.py` lookups**

`benchmarks/_common/make_tables.py:92-95`: `d.get("switch-plain")` → `d.get("split-plain")`, `_pick_residual(d, "switch", ...)` → `"split"`, `d.get("absolute-plain")` → `d.get("mixed-plain")`, `_pick_residual(d, "absolute", ...)` → `"mixed"`.

- [ ] **Step 4: Rename standalone scripts**

Apply `"absolute"`→`"mixed"`, `"switch"`→`"split"` at the cited lines in `deep_residual_run.py`, `deep_init_run.py`, `run.py`, `monoresidual_gate_{ablation,trap,scale}.py`, `loan_size_ladder_run.py`. (Display-label strings in `deep_init_run.py:31-33` like `"absolute (new init)"` → `"mixed (new init)"`.)

- [ ] **Step 5: Update benchmark tests**

`tests/benchmarks/test_search.py:55-58` flavor_name assertions → `"split-plain"`, `"mixed-residual"`, `"split-deep"`, `"mixed-deep"`; fixture `flavors=(("switch",...),)` and `mode=` args → new names.
`tests/benchmarks/test_search_cli.py:61,63-64`: `_parse_flavors("absolute-deep")==(("absolute",True,True),)` → `_parse_flavors("mixed-deep")==(("mixed",True,True),)`, etc.
`test_model_builder_{torch,jax,keras}.py`: parametrize `["switch","absolute"]` → `["split","mixed"]` + `Literal` annotations.
`test_search_spaces.py:16,38`, `test_config_io.py:12,22` (incl. assertion `cfg.mode == "switch"` → `== "split"`), `test_config.py`, `test_run_dataset.py`, `test_stage2_gate.py`, `test_deep_residual_run.py:` positional `"absolute"`→`"mixed"`, `test_init_diagnostics.py`, `test_runner.py`, `test_smoke.py`, `test_results.py`, `test_model_builder_keras.py` — apply the same substitutions.

- [ ] **Step 6: Run benchmark tests green**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks -v`
Expected: PASS. (Some benchmark tests are torch-oriented; run under torch.)

- [ ] **Step 7: Verify + commit**

```bash
git grep -nE '"(absolute|switch)"' benchmarks tests/benchmarks   # expect none (except inside result-file paths, handled in Task 7)
git add benchmarks tests/benchmarks
git commit --no-gpg-sign -m "refactor(benchmarks): rename mode/flavor absolute->mixed, switch->split"
```

---

### Task 7: Migrate committed benchmark result JSONs

**Files:**
- Rename + edit: `benchmarks/results/phase2/*.json` (30 files: filename + `"flavor"` field)
- Edit: `benchmarks/results/deep-residual/trainability.json` (38 `"mode"` values)
- Edit: `benchmarks/results/deep-init/trainability.json` (`"method"` display labels)
- Test: `tests/benchmarks/test_results.py` / `make_tables` resolution

**Interfaces:** Consumes the renamed `make_tables.py` lookups (Task 6). Produces result files whose flavor/mode strings match the new names so table generation resolves.

- [ ] **Step 1: Migration script (contents + phase2 filenames)**

Run:
```bash
python3 - <<'PY'
import json, pathlib
RES = pathlib.Path("benchmarks/results")
sub = {"absolute": "mixed", "switch": "split"}
def repl(s):
    for a,b in sub.items(): s = s.replace(a,b)
    return s
# phase2: rewrite "flavor" field and rename file
for p in sorted((RES/"phase2").glob("*.json")):
    d = json.loads(p.read_text())
    if "flavor" in d: d["flavor"] = repl(d["flavor"])
    p.write_text(json.dumps(d, indent=2) + "\n")
    new = p.with_name(repl(p.name))
    if new != p: p.rename(new)
# deep-residual: rewrite every "mode" value
for name in ["deep-residual/trainability.json", "deep-init/trainability.json"]:
    p = RES/name
    if not p.exists(): continue
    txt = p.read_text()
    d = json.loads(txt)
    def walk(o):
        if isinstance(o, dict):
            for k,v in o.items():
                if k in ("mode","method") and isinstance(v,str): o[k]=repl(v)
                else: walk(v)
        elif isinstance(o, list):
            for x in o: walk(x)
    walk(d)
    p.write_text(json.dumps(d, indent=2) + "\n")
print("done")
PY
```

- [ ] **Step 2: Verify no old names remain in results**

Run: `git grep -nE '(absolute|switch)' benchmarks/results` — expect none. Also confirm the 30 phase2 files were renamed: `ls benchmarks/results/phase2 | grep -E 'absolute|switch'` → empty.

- [ ] **Step 3: Confirm table generation still resolves the keys**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_results.py -v` (and any `make_tables` test). Expected: PASS. If there's no direct test, run the table builder used by the docs (`grep -rl make_tables benchmarks | head` to find the entrypoint) as a smoke check.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/results
git commit --no-gpg-sign -m "chore(benchmarks): migrate committed result JSONs to mixed/split names"
```

---

### Task 8: Update user-facing docs

**Files:**
- Modify: `README.md` (lines 17, 47, 100, 101, 139, 140)
- Modify: `docs/guides/index.md:18`, `docs/benchmarks/index.md:21,30`, `docs/benchmarks/protocol.md:85,88`, `docs/benchmarks/large-dataset-screen.md:3`
- Modify: `docs/concepts/monotonic-residual.md` (~18 refs), `docs/benchmarks/deep-residual-accuracy.md` (~37 refs)
- Note (defer): `docs/benchmarks/*.ipynb` (executed notebooks) — leave to the docs phase (spec §9), which regenerates them.

**Interfaces:** none (prose + code-example strings).

- [ ] **Step 1: Rename in markdown docs**

Apply `mode="absolute"`→`mode="mixed"`, `mode="switch"`→`mode="split"`, and backtick `` `absolute` ``→`` `mixed` ``, `` `switch` ``→`` `split` `` across the markdown files listed. Do **not** touch `docs/_build/**`, `docs/superpowers/**` (dated archives), or the `.ipynb` files.

- [ ] **Step 2: Verify markdown docs are clean**

Run: `git grep -nE 'mode="?(absolute|switch)|`(absolute|switch)`' README.md docs/guides docs/concepts docs/benchmarks --  ':!docs/_build' ':!*.ipynb'`
Expected: none.

- [ ] **Step 3: Docs build (warnings-as-errors)**

Run: `./tools/build-docs.sh` (or `uv run sphinx-build -W docs docs/_build/html`).
Expected: build succeeds with no warnings.

- [ ] **Step 4: Commit**

```bash
git add README.md docs
git commit --no-gpg-sign -m "docs: rename mode absolute->mixed, switch->split in user-facing docs"
```

---

### Task 9: Full-repo verification + PR

**Files:** none (verification only).

- [ ] **Step 1: Global grep — zero stray old mode literals**

Run: `git grep -nE '"(absolute|switch)"' -- ':!docs/superpowers' ':!docs/_build' ':!*.ipynb' ':!benchmarks/results'`
Expected: only `_RENAMED_MODES` keys in `mononet/core/config.py`. Investigate anything else.

- [ ] **Step 2: Full test suite, all backends**

```bash
MONONET_TEST_BACKEND=torch uv run pytest
MONONET_TEST_BACKEND=jax   uv run pytest
MONONET_TEST_BACKEND=keras uv run pytest
```
Expected: PASS (uninstalled backends skip).

- [ ] **Step 3: Lint, types, hooks**

```bash
uv run ruff check --exit-non-zero-on-fix
uv run ruff format --check
uv run mypy
uv run pre-commit run --all-files
```
Expected: all PASS (includes `reference-hash` and docs build).

- [ ] **Step 4: Open the PR**

```bash
git push -u origin spec/monotone-constructions-init-and-ablation
gh pr create --base main --title "refactor: rename construction mode absolute->mixed, switch->split (hard break)" \
  --body-file - <<'EOF'
Phase 1 of the monotone-constructions spec: pure rename of the `mode` values
`absolute`→`mixed` and `switch`→`split`, hard break (old names raise a
`ValueError` naming the replacement). No behavior change — equivalence vectors
are numerically identical (only the `"mode"` field and `REFERENCE_HASH` differ).

Covers core, all three backends, equivalence vectors, benchmarks (flavor
plumbing + committed result JSONs), tests, and user-facing docs. Adds the
`alternate` mode and the composition-aware init in later phases.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
```

---

## Notes for later phases (out of scope here)

- **Phase 2** adds `mode="alternate"` to the `Mode` literal + the composition-aware init via the `prev=` reference (spec §4, §7).
- **Optional cleanups deferred:** rename `absolute_init_params`→`mixed_init_params` and the `_absolute_default` attribute; rename equivalence case filename slugs (`-switch-`/`-abs-`); migrate `.ipynb` mode references (folded into the docs phase, spec §9).
