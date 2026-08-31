.PHONY: quality style test test-network test-slow test-all test-matrix serve-docs build-docs serve-docs-versioned

PACKAGE = src/trimbed
SOURCES = $(PACKAGE) tests/ examples/ docs/

quality:
	uv run ruff check $(SOURCES)
	uv run ruff format --check $(SOURCES)

style:
	uv run ruff check $(SOURCES) --fix
	uv run ruff format $(SOURCES)

# The offline suite. Marker selection and the coverage flags live in the addopts in
# pyproject.toml, so the targets below only say which half of the Hub tests they want.
test:
	uv run pytest

# The Hub tokenizer trims: network, but no weights. Coverage over a marker subset means
# nothing and would overwrite coverage.xml with a misleading figure, so switch it off.
test-network:
	uv run --all-extras pytest -m "network and not slow" --no-cov

# Downloads a real checkpoint per case and runs a forward pass over it. Minutes each.
test-slow:
	uv run --all-extras pytest -m "network and slow" --no-cov

# An empty -m overrides the one in addopts and deselects nothing.
test-all:
	uv run --all-extras pytest -m ""

# The same offline suite as `test`, but once per interpreter the CI matrix covers.
# Version-specific breakage (a stdlib fix that only landed in 3.13, a syntax feature 3.12
# lacks) is invisible in a single local venv and only shows up as a red job after
# pushing; this is the pre-push check for that.
#
# The version list is read out of the workflow itself so the two cannot drift. `tr` strips
# the quotes around each version, and any stray CR that survived a Windows checkout --
# .gitattributes should prevent the latter, but a lone CR here would silently yield an
# empty list rather than an error.
PY_VERSIONS ?= $(shell tr -d '\r"' < .github/workflows/ci.yml | sed -n 's/^ *python-version: \[\(.*\)\] *$$/\1/p' | tr ',' ' ')
# One venv per interpreter, kept next to (not on top of) the `.venv` that uv manages for
# everyday work. uv hardlinks from its cache, so these are cheap.
MATRIX_VENV_DIR ?= .venvs

test-matrix:
	@test -n "$(PY_VERSIONS)" || \
		{ echo "No python-version matrix found in .github/workflows/ci.yml"; exit 1; }
	@set -e; \
	for v in $(PY_VERSIONS); do \
		echo "==> Python $$v"; \
		UV_PROJECT_ENVIRONMENT=$(MATRIX_VENV_DIR)/py$$v \
			uv sync --locked --group dev --python "$$v" --quiet; \
		UV_PROJECT_ENVIRONMENT=$(MATRIX_VENV_DIR)/py$$v \
			uv run --no-sync --python "$$v" pytest --no-cov; \
	done

# Everyday local preview: live reload, no git branches involved.
serve-docs:
	uv run mkdocs serve

# What CI runs on every pull request. Fails on broken cross-references and nav entries.
build-docs:
	uv run mkdocs build --strict

# Only needed to exercise the mike version selector and the "outdated version" banner
# locally. Writes to a throwaway local branch; never push it.
DOCS_BRANCH ?= tmp-gh-pages
DOCS_VERSION ?= 0.0.0
DOCS_ALIAS ?= latest
DOCS_ADDR ?= 127.0.0.1:8000
DOCS_SOURCE_REF ?= main

serve-docs-versioned:
	DOCS_SOURCE_REF=$(DOCS_SOURCE_REF) uv run mike deploy --branch $(DOCS_BRANCH) --update-aliases $(DOCS_VERSION) $(DOCS_ALIAS)
	uv run mike set-default --branch $(DOCS_BRANCH) $(DOCS_ALIAS)
	uv run mike serve -b $(DOCS_BRANCH) -a $(DOCS_ADDR)
