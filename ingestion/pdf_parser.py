"""
DocuSense AI - PDF Parser
To extract test, metada, and page info from pdfs,
Uses PyMuPDF (fitz) for high quality extraction
"""

import fitz # PyMuPDF
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger

# Data Models

@dataclass
class Pagecontent:
    """to represent content from a single PDG page"""
    page_number: int
    text: str
    char_count: int
    has_images: bool
    image_count: int
    has_tables: bool
    width: float
    height: float
    rotation: int

@dataclass
class DocumentContent:
    """to represent fully parsed pdf document"""
    doc_id: str
    file_name: str
    file_path: str
    doc_type: str
    total_pages: int
    total_chars: int
    pages: list
    metadata: dict
    processing_status: str

#pdf parser class

class PDFParser:
    """
    To extract text and structure from pdf files,
    handles all scanned pdfs, multi-column layours, headers/footers, page numbers.
    """

    def __init__(self):
        self.min_text_length = 10  # we will skip pages with less than this

    def parse(self, pdf_path: Path, doc_type: str = "general") -> DocumentContent:
        """Main method -- returns document connect with all pages"""
        logger.info(f"Parsing PDF: {pdf_path.name}")

        try:
            doc = fitz.open(pdf_path)

            # to extract pdf-level metatadata
            pdf_metadata = self._extract_metadata(doc, pdf_path)

            # to extract content from each page
            pages =[]
            total_chars = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_content = self._parse_page(page, page_num + 1)
                pages.append(asdict(page_content))
                total_chars += page_content.char_count

            doc.close()

            # creating document content object
            doc_content = DocumentContent(
                doc_id=pdf_path.stem,
                file_name=pdf_path.name,
                file_path=str(pdf_path),
                doc_type=doc_type,
                total_pages=len(pages),
                total_chars=total_chars,
                pages=pages,
                metadata=pdf_metadata,
                processing_status="parsed"
            )

            logger.success(
                f"Parsed {pdf_path.name}:"
                f"{len(pages)} pages"
                f"{total_chars:,} chars"
            )

            return doc_content
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path.name}: error is {e}")
            raise

    def _parse_page(self, page: fitz.Page, page_num: int) -> Pagecontent:
        """to extract content from single page"""

        # page dimensions
        rect = page.rect
        width = rect.width
        height = rect.height

        # Extract text 
        text = self._extract_text_smart(page)

        # check for images
        images = page.get_images()
        has_images = len(images)>0

        # to detect potential tables (grid like patterns)
        has_tables = self._detect_table_presence(page)

        return Pagecontent(
            page_number=page_num,
            text=text,
            char_count=len(text),
            has_images=has_images,
            has_tables=has_tables,
            image_count=len(images),
            width=width,
            height=height,
            rotation=page.rotation
        )

    def _extract_text_smart(self, page: fitz.Page) -> str:
        """
        Smart text extraction to handle multi-column layours, headers footers and reading order
        """
        try:
            # text blocks with position information
            blocks = page.get_text("blocks")

            if not blocks:
                return ""

            # sorting blockes top to bottom then left to right
            blocks_sorted = sorted(blocks, key=lambda b: (round(b[1] / 20) * 20, b[0]))

            # fetch text from each block
            text_parts = []
            for block in blocks_sorted:
                if block in blocks_sorted:
                    if block[6] == 0:
                        block_text = block[4].strip()
                        if len(block_text) > self.min_text_length:
                            text_parts.append(block_text)

            full_text = "\n\n".join(text_parts)

            # clean up text
            full_text = self._clean_text(full_text)

            return full_text

        except Exception as e:
            logger.warning(f"Smart extraction failed, using simple: {e}")
            return page.get_text()


    def _clean_text(self, text: str) -> str:
        """clean extracted text"""
        import re
        # removing excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # removing pdf artifacts
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

        # leading and trailing white spaces
        text = text.strip()

        return text

    def _detect_table_presence(self, page: fitz.Page) -> bool:
        """
        to detect if pdf page has any tables by looking for line patterns"""

        try:
            # Look for horizontal and vertical lines
            drawings = page.get_drawings()

            h_lines = 0
            v_lines = 0

            for drawing in drawings:
                for item in drawing.get("items", []):
                    if item[0] == "l": # line
                        p1 = item[1]
                        p2 = item[2]
                        x0, y0 = p1.x, p1.y
                        x1, y1 = p2.x, p2.y

                        diff_x = x1-x0
                        diff_y = y1-y0

                        if (diff_y < 3 and diff_y > -3) and (diff_x > 50 or diff_x < -50):
                            h_lines +=1

                        if (diff_x < 3 and diff_x > -3) and (diff_y > 20 or diff_y < -20):
                            v_lines +=1

            return h_lines >= 3 and v_lines >=2

        except Exception as e:
            logger.warning(f"Failed to identify tables {e}")
            return False

    def _extract_metadata(self, doc: fitz.Document, pdf_path: Path) -> dict:
        """extract PDF metadata"""
        raw_meta = doc.metadata

        return{
            "title": raw_meta.get("title", pdf_path.stem),
            "author": raw_meta.get("author", "Unknown"),
            "subject": raw_meta.get("subject", ""),
            "creator": raw_meta.get("creator", ""),
            "creation_date": raw_meta.get("creationDate", ""),
            "page_count": len(doc),
            "file_size_bytes": pdf_path.stat().st_size,
            "file_name": pdf_path.name,
        }

    def save_parsed(self, doc_content: DocumentContent, output_dir: Path) -> Path:
        """Save parsed content to json"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{doc_content.doc_id}_parsed.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(doc_content), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved parsed content: {output_path.name}")
        return output_path

# Batch Parser

class BatchPDFParser:
    """parse multiple pdfs in batch"""

    def __init__(self):
        self.parser = PDFParser()

    def parse_directory(self, input_dir: Path, output_dir: Path, doc_type_map: dict = None) -> list:
        """parse all pdfs in a directory"""
        pdf_files = list(input_dir.glob("*.pdf"))

        if not pdf_files:
            logger.error(f"No PDFs found in {input_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} pdfs to parse")

        results = []
        failed = []

        for pdf_path in pdf_files:
            try:
                # Determine document type
                doc_type = "general"
                if doc_type_map:
                    doc_type = doc_type_map.get(pdf_path.stem, "general")
                elif "invoice" in pdf_path.stem.lower():
                    doc_type = "invoice"
                elif "financial" in pdf_path.stem.lower():
                    doc_type = "financial_report"
                elif "contract" in pdf_path.stem.lower():
                    doc_type = "contract"
                elif any(kw in pdf_path.stem.lower() for kw in ["paper", "bert", "llama", "attention", "rag"]):
                    doc_type = "research_paper"

                # parse 
                doc_content = self.parser.parse(pdf_path, doc_type)

                # save it
                self.parser.save_parsed(doc_content, output_dir)

                results.append({
                    "doc_id": doc_content.doc_id,
                    "file_name": doc_content.file_name,
                    "doc_type": doc_content.doc_type,
                    "total_pages": doc_content.total_pages,
                    "total_chars": doc_content.total_chars,
                    "status": "success"
                })

            except Exception as e:
                logger.error(f"failed: {pdf_path.name}: {e}")
                failed.append({
                    "file_name": pdf_path.name,
                    "error": str(e),
                    "status": "failed"
                })

        self._print_summary(results, failed)

        return results

    def _print_summary(self, results: list, failed: list):
        print("\n" + "="*60)
        print(" PDF Parsing summary")
        print("="*60)

        for r in results:
            print(f"\n {r['file_name']}")
            print(f"    type: {r['doc_type']}")
            print(f"    pages: {r['total_pages']}")
            print(f"    chars: {r['total_chars']}")

        if failed:
            print("\n FAILED:")
            for f in failed:
                print(f"{f['file_name']}: {f['error']}")

        print("\n" + "="*60)
        print(f"success = {len(results)}")
        print(f"failure = {len(failed)}")