# SPDX-License-Identifier: Apache-2.0
r"""sphinx-polyversion driver for mononet docs.

Builds the ``main`` branch (and any future ``v*.*.*`` tag) into
``docs/_build/html/<ref-name>/``. The PyData theme version switcher reads
``versions.json`` (generated separately by ``tools/gen_versions_json.py``).

The per-ref ref name is delivered to ``docs/conf.py`` via the
``POLYVERSION_DATA`` environment variable that ``sphinx-polyversion`` sets on
each ``sphinx-build`` invocation.

No per-ref Python environment is created: builds reuse the host environment
that invokes this driver (set up by ``uv sync --group docs --extra all``).
This is correct as long as docs dependencies stay the same across refs. Once
released tags pin specific dependency versions for their own docs builds,
switch ``env`` to ``Pip.factory(...)`` or an equivalent isolating env class.

Refs are enumerated from **local** ``refs/heads/*`` (branches) and
``refs/tags/*`` (tags), not from ``refs/remotes/origin/*``. ``sphinx-polyversion``
2.0's parser misclassifies remote-tracking refs (the ``\\w+`` alternative in its
ref regex greedily consumes ``"remotes"``), so we leave ``remote`` unset and
require the caller to have the desired branches/tags present locally. In CI,
``actions/checkout`` with ``fetch-depth: 0`` and ``fetch-tags: true`` gives us
``refs/heads/main`` and ``refs/tags/v*`` locally. For local dev, run
``git fetch --tags && git branch -f main origin/main`` before invoking this
driver.
"""

from __future__ import annotations

from pathlib import Path

from sphinx_polyversion import DefaultDriver
from sphinx_polyversion.environment import Environment
from sphinx_polyversion.git import Git, file_predicate
from sphinx_polyversion.sphinx import SphinxBuilder

#: Branches to build.
BRANCH_REGEX = r"^main$"

#: Tags to build (semver tags only).
TAG_REGEX = r"^v\d+\.\d+\.\d+$"

#: Build output dir, relative to repo root.
OUTPUT_DIR = "docs/_build/html"

#: Sphinx source dir, relative to each ref's worktree.
SOURCE_DIR = "docs"

#: Arguments forwarded to ``sphinx-build``.
SPHINX_ARGS = ["-W"]

src = Path(SOURCE_DIR)
root = Path(__file__).resolve().parent.parent

DefaultDriver(
    root=root,
    output_dir=OUTPUT_DIR,
    vcs=Git(
        branch_regex=BRANCH_REGEX,
        tag_regex=TAG_REGEX,
        buffer_size=1 * 10**9,  # 1 GB; large enough for the worktree exports
        predicate=file_predicate([src]),
    ),
    builder=SphinxBuilder(src, args=SPHINX_ARGS),
    env=Environment.factory(),
).run()
