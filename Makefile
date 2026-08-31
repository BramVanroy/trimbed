.PHONY: quality style test test-network test-slow test-all

quality:
	uv run ruff check src/trimbed tests/ examples/
	uv run ruff format --check src/trimbed tests/ examples/

style:
	uv run ruff check src/trimbed tests/ examples/ --fix
	uv run ruff format src/trimbed tests/ examples/

# The offline suite. Marker selection and the coverage flags live in the addopts in
# pyproject.toml, so the targets below only say which half of the Hub tests they want.
test:
	uv run pytest

# The Hub tokenizer trims: network, but no weights.
test-network:
	uv run --all-extras pytest -m "network and not slow"

# Downloads a real checkpoint per case and runs a forward pass over it. Minutes each.
test-slow:
	uv run --all-extras pytest -m "network and slow"

# An empty -m overrides the one in addopts and deselects nothing.
test-all:
	uv run --all-extras pytest -m ""
