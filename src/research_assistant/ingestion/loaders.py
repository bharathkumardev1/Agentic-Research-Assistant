"""Loaders that turn source files (and arXiv papers) into text + metadata.

Heavy / optional dependencies (``pypdf``, ``arxiv``) are imported lazily so
that the rest of the package can be used without them installed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Tuple

from ..schemas import PaperMetadata

LoadedPaper = Tuple[PaperMetadata, str]

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def _slugify(value: str, max_len: int = 48) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value[:max_len] or "paper"


def _make_paper_id(seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    return f"{_slugify(Path(seed).stem)}-{digest}"


def load_text_file(path: Path) -> LoadedPaper:
    """Load a ``.txt`` / ``.md`` file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = _title_from_text(text) or path.stem.replace("_", " ").title()
    meta = PaperMetadata(
        paper_id=_make_paper_id(str(path)),
        title=title,
        source=str(path),
    )
    return meta, text


def load_pdf_file(path: Path) -> LoadedPaper:
    """Load a PDF using :mod:`pypdf`."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Reading PDFs requires 'pypdf'. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())

    info = getattr(reader, "metadata", None)
    pdf_title = (getattr(info, "title", None) or "").strip() if info else ""
    title = pdf_title or _title_from_text(text) or path.stem.replace("_", " ").title()

    meta = PaperMetadata(
        paper_id=_make_paper_id(str(path)),
        title=title,
        source=str(path),
    )
    return meta, text


def load_file(path: Path) -> LoadedPaper:
    """Dispatch to the right loader based on file extension."""
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return load_pdf_file(path)
    if suffix in TEXT_SUFFIXES:
        return load_text_file(path)
    raise ValueError(f"Unsupported file type: {path.suffix} ({path})")


def load_paths(paths: List[Path]) -> List[LoadedPaper]:
    """Load every supported file in the given files and/or directories."""
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(
                sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES)
            )
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")

    papers: List[LoadedPaper] = []
    for f in files:
        if f.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        meta, text = load_file(f)
        if text.strip():
            papers.append((meta, text))
    return papers


def fetch_arxiv(
    query: str,
    max_results: int = 5,
    download_dir: Path = Path("data/raw"),
) -> List[LoadedPaper]:
    """Search arXiv, download the PDFs and load their full text.

    ``query`` may be a search string (e.g. ``"retrieval augmented generation"``)
    or an explicit arXiv id list passed via the ``id_list`` syntax handled by
    the ``arxiv`` package.
    """
    try:
        import arxiv
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Fetching from arXiv requires the 'arxiv' package. "
            "Install it with: pip install arxiv"
        ) from exc

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()

    papers: List[LoadedPaper] = []
    for result in client.results(search):
        short_id = result.get_short_id()
        filename = f"{_slugify(short_id)}.pdf"
        pdf_path = download_dir / filename
        if not pdf_path.exists():
            result.download_pdf(dirpath=str(download_dir), filename=filename)
        _, text = load_pdf_file(pdf_path)
        meta = PaperMetadata(
            paper_id=_slugify(short_id),
            title=result.title.strip(),
            authors=[a.name for a in result.authors],
            year=result.published.year if result.published else None,
            source=result.entry_id,
        )
        if text.strip():
            papers.append((meta, text))
    return papers


def _title_from_text(text: str) -> str:
    """Heuristically pull a title from the first meaningful line of text."""
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if len(line) >= 8 and not line.lower().startswith(("abstract", "http")):
            return line[:160]
    return ""
