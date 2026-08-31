"""Custom exceptions for trimbed."""

from __future__ import annotations


class MissingDependencyError(ImportError):
    """An optional dependency is required for the requested operation but is not installed.

    Subclasses `ImportError`, so `except ImportError` still catches it, while the
    message can name the extra to install.
    """

    def __init__(self, package: str, extra: str, purpose: str) -> None:
        """Build the error.

        Args:
            package: Import name of the missing package.
            extra: Name of the optional-dependency extra that provides it.
            purpose: Short description of what the package is needed for.
        """
        super().__init__(
            f"{purpose} requires the '{package}' package, which is not installed. "
            f"Install it with: pip install 'trimbed[{extra}]', or"
            f" with uv: uv add 'trimbed[{extra}]'."
        )
        self.package = package
        self.extra = extra
