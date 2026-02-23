"""LaTeX source file extraction — sends raw source to LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LaTeXDocument:
    """Extracted LaTeX document."""

    file_path: str
    full_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str = "latex"


class LaTeXExtractor:
    """Extract content from LaTeX source files.

    Since the LLM can read raw LaTeX, we just return the source text.
    We do minimal metadata extraction for the report header.
    """

    def extract(self, file_path: str | Path) -> LaTeXDocument:
        """Read LaTeX source and return it as-is for LLM analysis."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"LaTeX file not found: {file_path}")

        raw_text = file_path.read_text(encoding="utf-8", errors="replace")

        # Optionally append .bib content so the LLM can see bibliography
        bib_text = self._load_bibliography(file_path, raw_text)
        if bib_text:
            raw_text += f"\n\n% === Bibliography file content ===\n{bib_text}"

        return LaTeXDocument(
            file_path=str(file_path),
            full_text=raw_text,
        )

    def _load_bibliography(self, tex_path: Path, text: str) -> str:
        """Try to load .bib file content if referenced."""
        import re

        bib_match = re.search(r"\\bibliography\{([^}]*)\}", text)
        if not bib_match:
            return ""

        bib_name = bib_match.group(1)
        for ext in ("", ".bib"):
            bib_path = tex_path.parent / f"{bib_name}{ext}"
            if bib_path.exists():
                try:
                    return bib_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    logger.warning(f"Failed to read bibliography: {e}")

        return ""
