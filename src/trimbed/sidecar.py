"""Carrying the source repository's vocabulary-independent files into the output.

sentence-transformers models are what make this necessary. Their pooling and dense
modules live in files the trim itself never writes, and without them the output reopens
with default mean pooling and no error to say so.
"""

from __future__ import annotations

import shutil
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub

from trimbed._logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

# Files copied from the source repository by default
DEFAULT_SIDECAR_PATTERNS: tuple[str, ...] = (
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",  # older variant of ST config
    # 1_Pooling/, 2_Dense/, 3_Normalize/ ... the numbered sentence-transformers modules.
    "[0-9]_*/*",
)


def _list_source_files(source: str, revision: str | None) -> list[str]:
    """List every file in the source repository, as paths relative to its root.

    Args:
        source: Hub model id or local path.
        revision: Revision to list, for a Hub id.

    Returns:
        Relative POSIX paths, e.g. `["1_Pooling/config.json", "config.json",
        "config_sentence_transformers.json", "model.safetensors", "modules.json", ...]`
        for sentence-transformers/all-MiniLM-L6-v2.
    """
    directory = Path(source)
    if directory.is_dir():
        return sorted(str(path.relative_to(directory).as_posix()) for path in directory.rglob("*") if path.is_file())

    return sorted(huggingface_hub.HfApi().list_repo_files(source, revision=revision))


def _fetch(source: str, name: str, revision: str | None) -> Path:
    """Return a local path to a file in the source repository.

    The file is downloaded from the Hub when `source` is a model id rather than a
    directory on disk.

    Args:
        source: Hub model id or local path.
        name: Path of the file relative to the repository root, e.g.
            `"1_Pooling/config.json"`.
        revision: Revision to fetch, for a Hub id.

    Returns:
        The path to read from.
    """
    directory = Path(source)
    if directory.is_dir():
        return directory / name

    return Path(huggingface_hub.hf_hub_download(source, name, revision=revision))


def copy_sidecar_files(
    source: str,
    output_dir: str | Path,
    patterns: Sequence[str] = DEFAULT_SIDECAR_PATTERNS,
    revision: str | None = None,
) -> list[str]:
    """Copy the source repository's non-vocabulary files next to the trimmed artefacts.

    Args:
        source: Hub model id or local path the trim started from.
        output_dir: Directory the trimmed artefacts were written to.
        patterns: Glob patterns matched against repository-relative paths, e.g.
            `"modules.json"` and `"[0-9]_*/*"`.
        revision: Revision to copy from, for a Hub id.

    Returns:
        The relative paths that were copied, in order, e.g. `["1_Pooling/config.json",
        "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json"]`.
        Empty for a plain checkpoint, which keeps everything in the files the trim writes.
    """
    destination_root = Path(output_dir)
    copied_files: list[str] = []

    for name in _list_source_files(source, revision):
        if not any(fnmatch(name, pattern) for pattern in patterns):
            continue
        destination = destination_root / name
        if destination.exists():
            logger.warning(f"{name} was written by the trim; not overwriting it with the copy from {source!r}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_fetch(source, name, revision), destination)
        copied_files.append(name)

    if copied_files:
        logger.info(f"copied {len(copied_files):,} file(s) from {source!r}: {', '.join(copied_files)}")
    return copied_files
