"""Content identity of the installed package."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from swim2real import __version__
from swim2real.search.store import canonical_sha256


def package_identity() -> dict[str, Any]:
    """Hash the installed Python sources and packaged settings."""
    root = Path(__file__).parent
    sources = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"}
    }
    return {
        "package": "swim2real",
        "version": __version__,
        "source_sha256": canonical_sha256(sources),
        "source_files": sources,
    }
