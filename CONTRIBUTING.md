# Contributing to trimbed

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). One command gets you a
working environment with every development tool:

```bash
uv sync --locked --group dev
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

The second `pre-commit install` is not optional if you want the matrix check below to run.
It is a separate hook stage, so it needs its own registration once per clone.

Python 3.12 or 3.13. Not 3.14 yet: `datasets` pulls in `multiprocess`, which has no cp314
wheels and fails to import there, so `requires-python` caps at `<3.14`.

Runtime dependencies are capped at the next minor rather than the next major, because
trimbed works against internals of `transformers`, `tokenizers`, `skeletoken` and `torch`
that move on minor releases. Raising a cap is its own change, with `make test-all` behind
it, not something to fold into an unrelated pull request.

## The checks

```bash
make style         # ruff check --fix + ruff format
make quality       # the non-mutating equivalent
make test          # the offline suite, with coverage
make test-network  # + the Hub tokenizer trims
make test-slow     # + the Hub model trims; minutes per case
make test-all      # everything
make build-docs    # mkdocs build --strict
make serve-docs    # live preview on localhost
```

Run `make style`, `make test` and `make build-docs` before finishing any change.

The suite is offline by default and covers **100% of statements and branches**. Keep it
there. The gate is Codecov rather than a number in `pyproject.toml`: `codecov.yml` fails a
pull request whose project coverage drops more than 3% against the base branch, or whose
own changed lines are less than 90% covered. A handful of genuinely unreachable guards
carry `# pragma: no cover` with a comment saying which invariant makes them unreachable.
Prefer a real test over a new pragma.

`make test-matrix` runs the offline suite once per interpreter in the CI matrix. It reads
the version list out of `.github/workflows/ci.yml`, so the two cannot drift. It is also the
pre-push hook, which is why version-specific breakage is caught before it becomes a red job.

## Style

- Python 3.12+: `|` unions, `type` aliases, PEP 695 generics, `Self`.
- Concise Google-style docstrings on every public function, enforced by ruff's `D` rules.
- Type hints are documentation, not a contract. There is no mypy, no ruff `ANN` rules and
  no `py.typed`, and none of the three is wanted.
- Raise built-in exceptions. `MissingDependencyError` is the only custom one, and it
  subclasses `ImportError`.
- No `# noqa` in the source. A rule worth suppressing in more than one place is ignored in
  `pyproject.toml`, with the reason next to it.
- Comments explain *why*, not what.

## Documentation

The site is MkDocs Material with mkdocstrings. Guides live in `docs/*.md`; the API
reference is one stub per module under `docs/api/`, each holding a title and a single
`::: trimbed.<module>` directive, and each listed in `mkdocs.yml`'s `nav`.

Two rules keep it navigable:

- **Cross-reference trimbed's own symbols** as ``[`Name`][trimbed.module.Name]``. Sphinx
  roles (`` :class:`Name` ``) render as literal text on the site and are blocked by a
  pre-commit hook. External names (`save_pretrained`, `tokenizers.Tokenizer`) stay as
  plain backticks.
- **Document an attribute with a docstring**, not a `#:` comment. Griffe only picks up the
  former; the latter renders as nothing at all. Also blocked by a pre-commit hook.

Adding a module means adding its stub page and its nav entry in the same pass.
`tests/test_docs.py` fails if either is missing, and `mkdocs build --strict` fails on any
cross-reference that does not resolve.

`examples/` and `tests/` must be kept in sync with `src/` the same way.
`tests/test_examples.py` executes every example against tiny in-process fixtures, so a
stale example fails the suite rather than rotting quietly.

## Pull requests

CI runs on every pull request to `main`: pre-commit over the whole tree, a strict docs
build, and the offline suite on Python 3.12 and 3.13 with coverage uploaded to Codecov.
The Hub tests are a separate scheduled workflow, deliberately off the pull-request path so
that a Hub outage or a several-gigabyte download never blocks a merge. You can run them on
demand from the Actions tab.

## Releases

Versions come from git tags through hatch-vcs; nothing is hand-edited.

1. Tag the commit `vX.Y.Z` and push the tag.
2. Publish a GitHub Release for it.

That fires two workflows in parallel: `publish.yml` builds and uploads to PyPI through
trusted publishing, and `docs.yml` deploys the docs for that version with mike and moves
the `latest` alias. The tag must start with `v`, which the docs workflow checks.
