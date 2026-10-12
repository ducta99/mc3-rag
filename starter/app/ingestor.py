"""Defensive Multimodal Ingestor for AMD AI League - Match 3.

Handles heterogeneous file formats defensively:
- PDF (.pdf): pypdf with encrypted and corrupted file resilience.
- Word (.docx): Pure-python zip/XML parser (zero external deps).
- Excel (.xlsx): openpyxl sheet/row extraction.
- CSV (.csv): csv DictReader formatted rows.
- Code & Logs (.py, .log, .txt): UTF-8/Latin-1 resilient read.
- Images (.png, .jpg, .jpeg): RapidOCR / Tesseract / Vision fallback.
- Defensive traps:
  * Empty directories -> silently ignored.
  * Permission denied (chmod 000, dropped DAC_OVERRIDE) -> caught, logged, skipped.
  * Encrypted files -> caught, logged, skipped.
  * Unknown extensions -> skipped.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Extensions supported by the pipeline
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".csv",
    ".txt",
    ".log",
    ".py",
    ".png",
    ".jpg",
    ".jpeg",
}


@dataclass
class DocumentChunk:
    rel_path: str
    chunk_id: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class IngestedDocument:
    rel_path: str
    file_type: str
    raw_text: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


class DocumentParser:
    """Parses individual documents safely without raising unhandled exceptions."""

    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self._ocr_engine = None
        self._ocr_attempted = False

    def _get_ocr_engine(self):
        if not self.enable_ocr or self._ocr_attempted:
            return self._ocr_engine
        self._ocr_attempted = True
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr_engine = RapidOCR()
            logger.info("Initialized RapidOCR engine successfully.")
        except Exception as e:
            logger.warning("RapidOCR not available (%s). Trying pytesseract...", e)
            try:
                import pytesseract
                from PIL import Image
                # quick test
                self._ocr_engine = "pytesseract"
                logger.info("Initialized pytesseract fallback.")
            except Exception as e2:
                logger.warning("OCR engines unavailable (%s). Images will have empty text.", e2)
                self._ocr_engine = None
        return self._ocr_engine

    def parse_pdf(self, file_path: Path, rel_path: str) -> IngestedDocument:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            if reader.is_encrypted:
                try:
                    res = reader.decrypt("")
                    if res == 0:
                        return IngestedDocument(
                            rel_path=rel_path,
                            file_type=".pdf",
                            raw_text="",
                            error="File encrypted with password",
                        )
                except Exception as e:
                    return IngestedDocument(
                        rel_path=rel_path,
                        file_type=".pdf",
                        raw_text="",
                        error=f"File encrypted: {e}",
                    )

            pages_text = []
            chunks = []
            for page_idx, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    text = text.strip()
                    if text:
                        pages_text.append(f"--- Page {page_idx + 1} ---\n{text}")
                        chunks.append(
                            DocumentChunk(
                                rel_path=rel_path,
                                chunk_id=f"{rel_path}:p{page_idx + 1}",
                                content=text,
                                metadata={"page": str(page_idx + 1)},
                            )
                        )
                except Exception as page_err:
                    logger.warning("Error extracting page %d of %s: %s", page_idx, rel_path, page_err)

            full_text = "\n\n".join(pages_text)
            metadata = {}
            if "withdrawn" in full_text.lower() or "withdrawn" in rel_path.lower():
                metadata["status"] = "WITHDRAWN"
            if "supersedes" in full_text.lower():
                metadata["status"] = "SUPERSEDES"

            return IngestedDocument(
                rel_path=rel_path,
                file_type=".pdf",
                raw_text=full_text,
                chunks=chunks,
                metadata=metadata,
            )
        except Exception as e:
            return IngestedDocument(
                rel_path=rel_path,
                file_type=".pdf",
                raw_text="",
                error=f"PDF extraction error: {e}",
            )

    def parse_docx(self, file_path: Path, rel_path: str) -> IngestedDocument:
        """Pure standard-library parser for DOCX files (XML extraction)."""
        try:
            with zipfile.ZipFile(str(file_path), "r") as docx_zip:
                if "word/document.xml" not in docx_zip.namelist():
                    return IngestedDocument(
                        rel_path=rel_path,
                        file_type=".docx",
                        raw_text="",
                        error="Not a valid docx: missing word/document.xml",
                    )
                xml_data = docx_zip.read("word/document.xml")
                root = ET.fromstring(xml_data)

                # Word XML namespaces
                ns = {
                    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                }

                paragraphs = []
                chunks = []
                p_idx = 0

                # Extract paragraphs and table cells
                for elem in root.iter():
                    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                    if tag == "p":
                        texts = [t.text for t in elem.findall(".//w:t", ns) if t.text]
                        p_text = "".join(texts).strip()
                        if p_text:
                            paragraphs.append(p_text)
                            p_idx += 1
                            chunks.append(
                                DocumentChunk(
                                    rel_path=rel_path,
                                    chunk_id=f"{rel_path}:p{p_idx}",
                                    content=p_text,
                                )
                            )

                full_text = "\n".join(paragraphs)
                return IngestedDocument(
                    rel_path=rel_path,
                    file_type=".docx",
                    raw_text=full_text,
                    chunks=chunks,
                )
        except Exception as e:
            return IngestedDocument(
                rel_path=rel_path,
                file_type=".docx",
                raw_text="",
                error=f"DOCX extraction error: {e}",
            )

    def parse_xlsx(self, file_path: Path, rel_path: str) -> IngestedDocument:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(file_path), data_only=True)
            all_lines = []
            chunks = []

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                sheet_lines = [f"=== Sheet: {sheet_name} ==="]
                rows = list(ws.iter_rows(values_only=True))
                if not rows:
                    continue

                # Detect header if first row has string values
                header = [str(col).strip() if col is not None else f"col_{i}" for i, col in enumerate(rows[0])]
                sheet_lines.append("Columns: " + " | ".join(header))

                for r_idx, row in enumerate(rows[1:], start=2):
                    if all(v is None for v in row):
                        continue
                    row_pairs = []
                    for h, v in zip(header, row):
                        if v is not None:
                            row_pairs.append(f"{h}: {v}")
                    row_str = " | ".join(row_pairs)
                    if row_str:
                        sheet_lines.append(row_str)
                        chunks.append(
                            DocumentChunk(
                                rel_path=rel_path,
                                chunk_id=f"{rel_path}:{sheet_name}:r{r_idx}",
                                content=f"[{sheet_name}] {row_str}",
                                metadata={"sheet": sheet_name, "row": str(r_idx)},
                            )
                        )

                all_lines.append("\n".join(sheet_lines))

            full_text = "\n\n".join(all_lines)
            return IngestedDocument(
                rel_path=rel_path,
                file_type=".xlsx",
                raw_text=full_text,
                chunks=chunks,
            )
        except Exception as e:
            return IngestedDocument(
                rel_path=rel_path,
                file_type=".xlsx",
                raw_text="",
                error=f"XLSX extraction error: {e}",
            )

    def parse_csv(self, file_path: Path, rel_path: str) -> IngestedDocument:
        try:
            with open(str(file_path), "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                lines = []
                chunks = []
                for idx, row in enumerate(reader, start=1):
                    row_str = " | ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
                    lines.append(row_str)
                    chunks.append(
                        DocumentChunk(
                            rel_path=rel_path,
                            chunk_id=f"{rel_path}:r{idx}",
                            content=row_str,
                            metadata={"row": str(idx)},
                        )
                    )

            full_text = "\n".join(lines)
            return IngestedDocument(
                rel_path=rel_path,
                file_type=".csv",
                raw_text=full_text,
                chunks=chunks,
            )
        except Exception as e:
            return IngestedDocument(
                rel_path=rel_path,
                file_type=".csv",
                raw_text="",
                error=f"CSV extraction error: {e}",
            )

    def parse_text(self, file_path: Path, rel_path: str, ext: str) -> IngestedDocument:
        try:
            with open(str(file_path), "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            chunks = []
            # Split into meaningful paragraph or line chunks
            lines = content.splitlines()
            buffer = []
            for i, line in enumerate(lines, start=1):
                buffer.append(line)
                if len(buffer) >= 5 or line.strip() == "":
                    chunk_text = "\n".join(buffer).strip()
                    if chunk_text:
                        chunks.append(
                            DocumentChunk(
                                rel_path=rel_path,
                                chunk_id=f"{rel_path}:L{i}",
                                content=chunk_text,
                            )
                        )
                    buffer = []
            if buffer:
                chunk_text = "\n".join(buffer).strip()
                if chunk_text:
                    chunks.append(
                        DocumentChunk(
                            rel_path=rel_path,
                            chunk_id=f"{rel_path}:end",
                            content=chunk_text,
                        )
                    )

            return IngestedDocument(
                rel_path=rel_path,
                file_type=ext,
                raw_text=content,
                chunks=chunks,
            )
        except Exception as e:
            return IngestedDocument(
                rel_path=rel_path,
                file_type=ext,
                raw_text="",
                error=f"Text read error: {e}",
            )

    def parse_image(self, file_path: Path, rel_path: str, ext: str) -> IngestedDocument:
        engine = self._get_ocr_engine()
        if engine is None:
            return IngestedDocument(
                rel_path=rel_path,
                file_type=ext,
                raw_text="",
                error="No OCR engine available",
            )

        try:
            extracted_lines = []
            if hasattr(engine, "__call__") or hasattr(engine, "ocr"):
                # RapidOCR
                result, _ = engine(str(file_path))
                if result:
                    for item in result:
                        # item: [box, text, score]
                        if len(item) >= 2 and item[1]:
                            extracted_lines.append(str(item[1]).strip())
            elif engine == "pytesseract":
                import pytesseract
                from PIL import Image
                img = Image.open(str(file_path))
                text = pytesseract.image_to_string(img)
                extracted_lines = [l.strip() for l in text.splitlines() if l.strip()]

            full_text = "\n".join(extracted_lines)
            chunks = [
                DocumentChunk(
                    rel_path=rel_path,
                    chunk_id=f"{rel_path}:ocr",
                    content=full_text,
                )
            ] if full_text else []

            return IngestedDocument(
                rel_path=rel_path,
                file_type=ext,
                raw_text=full_text,
                chunks=chunks,
                metadata={"ocr": "true"},
            )
        except Exception as e:
            return IngestedDocument(
                rel_path=rel_path,
                file_type=ext,
                raw_text="",
                error=f"Image OCR error: {e}",
            )

    def parse_file(self, file_path: Path, corpus_root: Path) -> Optional[IngestedDocument]:
        """Parses a single file with total error containment."""
        try:
            rel_path = file_path.relative_to(corpus_root).as_posix()
        except Exception:
            rel_path = file_path.name

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping unsupported extension: %s (%s)", ext, rel_path)
            return None

        # Check permissions explicitly
        try:
            if not os.access(str(file_path), os.R_OK):
                logger.warning("File not readable (permission denied): %s", rel_path)
                return IngestedDocument(
                    rel_path=rel_path,
                    file_type=ext,
                    raw_text="",
                    error="Permission denied (unreadable)",
                )
        except Exception as e:
            logger.warning("Error checking permissions for %s: %s", rel_path, e)

        try:
            if ext == ".pdf":
                return self.parse_pdf(file_path, rel_path)
            elif ext == ".docx":
                return self.parse_docx(file_path, rel_path)
            elif ext == ".xlsx":
                return self.parse_xlsx(file_path, rel_path)
            elif ext == ".csv":
                return self.parse_csv(file_path, rel_path)
            elif ext in {".txt", ".log", ".py"}:
                return self.parse_text(file_path, rel_path, ext)
            elif ext in {".png", ".jpg", ".jpeg"}:
                return self.parse_image(file_path, rel_path, ext)
        except PermissionError as pe:
            logger.warning("PermissionError accessing %s: %s", rel_path, pe)
            return IngestedDocument(
                rel_path=rel_path,
                file_type=ext,
                raw_text="",
                error="PermissionError",
            )
        except Exception as ex:
            logger.warning("Unhandled error parsing %s: %s", rel_path, ex)
            return IngestedDocument(
                rel_path=rel_path,
                file_type=ext,
                raw_text="",
                error=str(ex),
            )
        return None


def walk_and_ingest_corpus(corpus_root: Path, enable_ocr: bool = True) -> List[IngestedDocument]:
    """Walks the corpus tree defensively and parses all valid files."""
    parser = DocumentParser(enable_ocr=enable_ocr)
    ingested_docs: List[IngestedDocument] = []

    if not corpus_root.exists() or not corpus_root.is_dir():
        logger.error("Corpus root does not exist or is not a directory: %s", corpus_root)
        return ingested_docs

    # Walk directory tree
    for root_dir, dirs, files in os.walk(str(corpus_root)):
        root_path = Path(root_dir)
        # Empty directories are naturally skipped when files is empty
        for file_name in files:
            file_path = root_path / file_name
            try:
                doc = parser.parse_file(file_path, corpus_root)
                if doc is not None:
                    ingested_docs.append(doc)
            except Exception as e:
                # Top level defensive catch: one bad file must NEVER crash the walk
                logger.error("Unexpected failure on file %s: %s", file_path, e)

    return ingested_docs
