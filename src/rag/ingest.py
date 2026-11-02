"""Extract policy text and group it into document-section chunks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import unicodedata

import pymupdf

from src.config import DATA_DIR


POLICY_PDF_PATH = DATA_DIR / "store_policies.pdf"
POLICY_SOURCE = "data/store_policies.pdf"
_MAJOR_HEADING = re.compile(r"^(\d{1,2})\.\s+(.+)$")
_PAGE_FOOTER = re.compile(r"^pagina\s+\d+$")


class PolicyDocumentError(ValueError):
    """Raised when the store policy PDF cannot be read or has no usable text."""


@dataclass(frozen=True, slots=True)
class PolicyLine:
    """One extracted line and its major-section heading, when identified."""

    text: str
    section_title: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyPage:
    """Extracted lines for a PDF page, retaining its one-based page number."""

    page_number: int
    lines: tuple[PolicyLine, ...]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass(frozen=True, slots=True)
class PolicyChunk:
    """A section-aware retrieval unit with deterministic source metadata."""

    chunk_id: str
    section: str | None
    page_start: int
    page_end: int
    pages: tuple[int, ...]
    text: str
    source: str


def _heading_title(line_text: str, spans: list[dict]) -> str | None:
    match = _MAJOR_HEADING.match(line_text.strip())
    if match is None:
        return None
    font_size = max((float(span.get("size", 0)) for span in spans), default=0)
    bold = any(
        int(span.get("flags", 0)) & 16
        or "bold" in str(span.get("font", "")).casefold()
        for span in spans
    )
    # In this PDF the major numbered headings are bold 14 pt; numbered body
    # steps are regular 10 pt and subsection headings are bold 12 pt.
    if font_size < 13 or not bold:
        return None
    return f"{match.group(1)}. {match.group(2).strip()}"


def extract_policy_pages(pdf_path: str | Path = POLICY_PDF_PATH) -> list[PolicyPage]:
    """Extract usable text from every PDF page and preserve its page number."""
    path = Path(pdf_path)
    if not path.is_file():
        raise PolicyDocumentError(f"Policy PDF not found: {path}")

    try:
        with pymupdf.open(str(path)) as document:
            if document.page_count == 0:
                raise PolicyDocumentError(f"Policy PDF contains no pages: {path}")

            pages: list[PolicyPage] = []
            for page_number, page in enumerate(document, start=1):
                extracted_lines: list[PolicyLine] = []
                page_dict = page.get_text("dict", sort=True)
                for block in page_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        text = "".join(str(span.get("text", "")) for span in spans)
                        text = text.replace("\u200b", "").strip()
                        if text:
                            extracted_lines.append(
                                PolicyLine(
                                    text=text,
                                    section_title=_heading_title(text, spans),
                                )
                            )
                pages.append(PolicyPage(page_number=page_number, lines=tuple(extracted_lines)))
    except PolicyDocumentError:
        raise
    except Exception as exc:
        raise PolicyDocumentError(f"Could not read policy PDF '{path}': {exc}") from exc

    if not any(page.text.strip() for page in pages):
        raise PolicyDocumentError(f"Policy PDF contains no usable text: {path}")
    return pages


def _normalize_line(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _is_repeated_page_furniture(text: str) -> bool:
    normalized = " ".join(_normalize_line(text).split())
    return normalized == "emporio da musica manual de politicas e procedimentos" or bool(
        _PAGE_FOOTER.fullmatch(normalized)
    )


def chunk_policy_pages(
    pages: list[PolicyPage],
    *,
    source: str = POLICY_SOURCE,
) -> list[PolicyChunk]:
    """Create one chunk per major section, retaining actual section page ranges."""
    chunks: list[PolicyChunk] = []
    current_section: str | None = None
    current_lines: list[str] = []
    current_pages: list[int] = []

    def flush() -> None:
        nonlocal current_lines, current_pages
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            current_pages = []
            return
        page_numbers = tuple(sorted(set(current_pages)))
        identity = "\0".join(
            (source, current_section or "", ",".join(map(str, page_numbers)), text)
        )
        chunk_id = f"policy-{sha256(identity.encode('utf-8')).hexdigest()[:16]}"
        chunks.append(
            PolicyChunk(
                chunk_id=chunk_id,
                section=current_section,
                page_start=page_numbers[0],
                page_end=page_numbers[-1],
                pages=page_numbers,
                text=text,
                source=source,
            )
        )
        current_lines = []
        current_pages = []

    for page in pages:
        for line in page.lines:
            if _is_repeated_page_furniture(line.text):
                continue
            if line.section_title is not None:
                flush()
                current_section = line.section_title
            if line.text.strip():
                current_lines.append(line.text.strip())
                current_pages.append(page.page_number)

    flush()
    if not chunks:
        raise PolicyDocumentError("Policy PDF contains no usable policy chunks.")
    return chunks


def extract_policy_chunks(pdf_path: str | Path = POLICY_PDF_PATH) -> list[PolicyChunk]:
    """Extract and section-chunk the configured policy PDF."""
    path = Path(pdf_path)
    try:
        source = path.resolve().relative_to(DATA_DIR.parent.resolve()).as_posix()
    except ValueError:
        source = path.as_posix()
    return chunk_policy_pages(extract_policy_pages(path), source=source)
