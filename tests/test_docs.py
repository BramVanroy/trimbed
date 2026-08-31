"""The API reference has to track `src/`, and nothing in mkdocs can notice a missing page.

`validation.nav.omitted_files` catches a page that is not in the nav, but a brand new
module with no page at all builds cleanly and is simply absent from the site, which is
how an API reference rots. These tests close that gap, the same way `test_examples.py`
keeps `examples/` from drifting.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "trimbed"
DOCS_ROOT = REPOSITORY_ROOT / "docs"
API_ROOT = DOCS_ROOT / "api"
MKDOCS_CONFIG = REPOSITORY_ROOT / "mkdocs.yml"

DIRECTIVE = re.compile(r"^::: (?P<target>[\w.]+)$", re.MULTILINE)


def _documented_modules() -> dict[str, Path]:
    """Map every module named by a `:::` directive to the page that names it."""
    return {
        match["target"]: page
        for page in sorted(API_ROOT.rglob("*.md"))
        for match in DIRECTIVE.finditer(page.read_text(encoding="utf-8"))
    }


def _defines_something(path: Path) -> bool:
    """Return whether a module defines a public class or function of its own.

    A package `__init__` that only re-exports has nothing to render: every symbol it
    names is documented on the module that defines it, and a second page for it would
    give each of them a duplicate anchor.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_")
        for node in tree.body
    )


def _public_modules() -> list[str]:
    """List the importable names of every public module that needs a reference page."""
    names = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            if not _defines_something(path):
                continue
            parts = parts[:-1]
        # `_logging` is private, but `cli.__main__` is the router and is documented.
        if any(part.startswith("_") and part != "__main__" for part in parts):
            continue
        names.append(".".join(parts))
    return names


def _nav_pages(node: object) -> list[str]:
    """Collect every page path the nav refers to, at any nesting depth."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [page for item in node for page in _nav_pages(item)]
    if isinstance(node, dict):
        return [page for value in node.values() for page in _nav_pages(value)]
    return []


@pytest.mark.parametrize("module", _public_modules())
def test_every_public_module_has_an_api_page(module):
    documented = _documented_modules()
    assert module in documented, f"no docs/api page renders {module}; add one and list it in mkdocs.yml"


@pytest.mark.parametrize("page", sorted(p.relative_to(DOCS_ROOT).as_posix() for p in API_ROOT.rglob("*.md")))
def test_every_api_page_is_in_the_nav(page):
    nav = yaml.safe_load(MKDOCS_CONFIG.read_text(encoding="utf-8"))["nav"]
    assert page in _nav_pages(nav), f"{page} exists but no mkdocs.yml nav entry points at it"


def test_api_pages_only_render_this_package():
    for target, page in _documented_modules().items():
        assert target == "trimbed" or target.startswith("trimbed."), f"{page} renders {target}, which is not trimbed"
