"""PDF text and metadata extraction using docling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PDFDocument:
    """Extracted PDF document with text and metadata."""

    file_path: str
    full_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str = "pdf"


class PDFExtractor:
    """Extract text and metadata from PDF files using docling."""

    def extract(self, file_path: str | Path) -> PDFDocument:
        """Extract text and metadata from a PDF file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        full_text = self._extract_text(file_path)
        metadata = self._extract_metadata(file_path)

        return PDFDocument(
            file_path=str(file_path),
            full_text=full_text,
            metadata=metadata,
        )

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from PDF using docling, with pypdf fallback."""
        # Primary: docling
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(file_path))
            text = result.document.export_to_markdown()
            if text and text.strip():
                return text
        except ImportError:
            logger.warning("docling not installed, falling back to pypdf")
        except Exception as e:
            logger.warning(f"docling extraction failed: {e}, falling back to pypdf")

        # Fallback: pypdf
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            raise RuntimeError(
                "No PDF extraction library available. Install docling or pypdf:\n"
                "  pip install docling   # recommended\n"
                "  pip install pypdf     # lightweight fallback"
            )

    def _extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract PDF metadata fields using pypdf."""
        metadata: dict[str, Any] = {}
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            info = reader.metadata
            if info:
                metadata = {
                    "title": info.get("/Title", ""),
                    "author": info.get("/Author", ""),
                    "creator": info.get("/Creator", ""),
                    "producer": info.get("/Producer", ""),
                    "subject": info.get("/Subject", ""),
                }
                metadata = {k: (v or "") for k, v in metadata.items()}
        except ImportError:
            logger.debug("pypdf not installed, skipping metadata extraction")
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")

        return metadata
