"""MkDocs hooks: clean up doctest examples and point source links at the built tag.

Docstrings write their examples with `>>>` prompts, which is the form a reader
recognises and the form `pytest --doctest-modules` would run if it were ever
switched on. mkdocstrings renders such a block as uncopyable console text, so
this hook post-processes the rendered HTML: it strips the `>>>` and `...`
prompts, turns expected-output lines into comments, and re-highlights the
result as Python.

It also rewrites the API source links. The templates in
`docs/overrides/python/material/` emit a bare repo-relative path; the
repository URL comes from `repo_url` in mkdocs.yml and the Git ref from the
`DOCS_SOURCE_REF` environment variable, which `.github/workflows/docs.yml`
sets to the release tag. Versioned docs therefore link at their own tag rather
than at whatever `main` happens to be.

The source files are never modified: only the rendered HTML differs.
"""

from __future__ import annotations

import html as html_lib
import os
import re
from typing import TYPE_CHECKING
from urllib.parse import quote

from pygments import highlight as pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page


_lexer = PythonLexer()
_formatter = HtmlFormatter(nowrap=True)

# Anchors emitted by the custom mkdocstrings templates, carrying a path that is
# relative to the repository root.
_SOURCE_LINK_PATTERN = re.compile(
    r'(?P<prefix><a\b[^>]*\bdata-source-link="github"[^>]*\bhref=")'
    r'(?P<source>[^"]+)'
    r'(?P<suffix>"[^>]*>)'
)

# mkdocstrings renders an Examples section as either a pycon or a plain text
# block, depending on how the docstring was written.
_DOCTEST_BLOCK_PATTERN = re.compile(
    r'<div class="language-(?:pycon|text) highlight">'
    r"<pre[^>]*><span></span><code>"
    r"(?P<code>.*?)"
    r"</code></pre></div>",
    flags=re.DOTALL,
)


def _strip_doctest_prompts(code: str) -> str:
    """Strip `>>>` and `...` prompts and turn expected output into comments.

    Args:
        code: The plain text of one rendered doctest block.

    Returns:
        The same code without prompts, with each expected-output line rewritten
        as a `#` comment so the block stays valid, copyable Python.
    """
    lines = code.split("\n")
    result: list[str] = []
    expect_output = False

    for line in lines:
        if line.startswith(">>> "):
            result.append(line[4:])
            expect_output = True
        elif line == ">>>":
            result.append("")
            expect_output = False
        elif line.startswith("... "):
            result.append(line[4:])
        elif line == "...":
            result.append("")
        elif expect_output and line != "":
            result.append("# " + line)
            expect_output = False
        else:
            result.append(line)
            if line.strip():
                expect_output = False

    return "\n".join(result).strip()


def _rewrite_doctest_block(match: re.Match[str]) -> str:
    """Rewrite a doctest-like code block into plain Python-highlighted code."""
    plain_code = html_lib.unescape(re.sub(r"<[^>]+>", "", match.group("code")))
    if ">>>" not in plain_code:
        return match.group(0)

    inner = pyg_highlight(_strip_doctest_prompts(plain_code), _lexer, _formatter)
    return '<div class="language-python highlight"><pre><span></span><code>' + inner + "</code></pre></div>"


def _rewrite_source_links(html: str, repo_url: str) -> str:
    """Point API source links at the Git ref the docs were built from."""
    git_ref = os.environ.get("DOCS_SOURCE_REF", "").strip() or "main"

    def transform_link(match: re.Match[str]) -> str:
        filepath, hash_sep, fragment = html_lib.unescape(match.group("source")).partition("#")
        url = f"{repo_url}/blob/{quote(git_ref, safe='/')}/{quote(filepath, safe='/')}"
        if hash_sep:
            url += f"#{fragment}"
        return f"{match.group('prefix')}{html_lib.escape(url, quote=True)}{match.group('suffix')}"

    return _SOURCE_LINK_PATTERN.sub(transform_link, html)


def on_page_content(html: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    """Post-process one rendered page: clean doctests, resolve source links.

    Args:
        html: The rendered HTML of the page.
        page: The page being rendered. Part of the MkDocs hook signature.
        config: The loaded mkdocs.yml, read for `repo_url`.
        files: Every file in the build. Part of the MkDocs hook signature.

    Returns:
        The rewritten HTML, or the input unchanged when no `repo_url` is set.
    """
    html = _DOCTEST_BLOCK_PATTERN.sub(_rewrite_doctest_block, html)

    repo_url = str(config.repo_url or "").rstrip("/").removesuffix(".git")
    if not repo_url:
        return html

    return _rewrite_source_links(html, repo_url)
