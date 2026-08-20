""" table extractor to extract tabled for pdf uses pdf plumber and convert table to text"""

import json
import pdfplumber
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger

# data models
@dataclass
class ExtractedTable:
    """for a single extracted table"""
    table_id: str
    doc_id: str
    page_number: int
    table_index: int
    headers: list
    rows: list
    raw_text: str
    markdown_text: str
    row_count: int
    col_count: int

# table Extractor class

class TableExtractor:
    """
    Extract tables from pdfs using pdf plumber and convert to multiple formats for RAG retrieval
    """

    def __init__(self):
        self.min_rows = 2
        self.min_cols = 2

    def extract_from_pdf(self, pdf_path: Path) -> list:
        """extract all tables and return list of ExtractedTable object"""
        logger.info(f"Extracting tables from: {pdf_path.name}")
        doc_id = pdf_path.stem
        all_tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):

                    tables = page.extract_tables()

                    if not tables:
                        continue

                    for table_idx, raw_table in enumerate(tables):

                        # clean and validate table
                        cleaned = self._clean_table(raw_table)

                        if not self._is_valid_table(cleaned):
                            continue

                        # parse table structure
                        headers, rows = self._parse_table_structure(cleaned)

                        markdown = self._to_markdown(headers, rows)
                        raw_text = self._to_text(headers, rows)
                        table = ExtractedTable(
                             table_id=f"{doc_id}_p{page_num}_t{table_idx}",
                             doc_id=doc_id,
                             page_number=page_num,
                             table_index=table_idx,
                             headers=headers,
                             rows=rows,
                             raw_text=raw_text,
                             markdown_text=markdown,
                             row_count=len(rows),
                             col_count=len(headers) if headers else 0
                        )

                        all_tables.append(table)
                        logger.info(f" table found: page {page_num}, {len(rows)} rows x {len(headers)} cols")

        except Exception as e:
            logger.error(f"Extracted {len(all_tables)} tables from {pdf_path.name}")

        logger.success(f"Extracted {len(all_tables)} tables from {pdf_path.name}")

        return all_tables

    def _clean_table(self, raw_table: list) -> list:
        """clean raw table data"""
        cleaned = []
        for row in raw_table:
            if row is None:
                continue

            # replacing None cells with empty string
            cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
            # skip completely empty rows
            if any(cell for cell in cleaned_row):
                cleaned.append(cleaned_row)

        return cleaned

    def _is_valid_table(self, table: list) -> bool:
        """Check if table has enough content"""
        if not table:
            return False
        if len(table) < self.min_rows:
            return False
        if not table[0]:
            return False
        if len(table[0]) < self.min_cols:
            return False
        return True

    def _parse_table_structure(self, table: list) -> tuple:
        """Split table into headers and rows and First row headers"""
        if not table:
            return [], []

        # use first row as header

        headers = table[0]
        rows = table[1:] if len(table) > 1 else []

        return headers, rows

    def _to_markdown(self, headers: list, rows: list) -> str:
        """Convert table to markdown format"""
        if not headers:
            return ""

        lines = []

        # header 
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        lines.append(header_line)

        # separator
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        lines.append(separator)

        # data rows
        for row in rows:
            # pad row if needed
            padded = list(row) + [""] * (len(headers) - len(row))
            row_line = "| "+" | ".join(str(cell) for cell in padded[:len(headers)])
            lines.append(row_line)

        return "\n".join(lines)

    def _to_text(self, headers: list, rows: list) -> str:
        """Convert table to natural language text for embedding and retrieval"""
        if not headers:
            return ""

        lines = []
        lines.append(f"Table with columns: {', '.join(str(h) for h in headers)}")
        lines.append("")

        for i, row in enumerate(rows):
            padded = list(row) + [""] * (len(headers) - len(row))
            parts = []
            for header, cell in zip(headers, padded):
                if cell:
                    parts.append(f"{header}: {cell}")
            if parts:
                lines.append(f"Row {i+1}: "+", ".join(parts))

        return "\n".join(lines)

    def save_tables(self, tables: list, output_dir: Path, doc_id:str) -> Path:
        """save extracted tables to json"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{doc_id}_tables.json"

        tables_data = [asdict(t) for t in tables]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tables_data, f , indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(tables)} tables to {output_path.name}")
        return output_path


# batch table Extractor

class BatchTableExtractor:
    """Extract tables from multiple pdfs"""

    def __init__(self):
        self.extractor = TableExtractor()

    def extract_directory(self, input_dir: Path, output_dir: Path) -> dict:
        """Extract tables frol all pdfs in a directory"""

        pdf_files = list(input_dir.glob("*.pdf"))
        all_results = {}

        for pdf_path in pdf_files:
            tables = self.extractor.extract_from_pdf(pdf_path)

            if tables:
                self.extractor.save_tables(tables, output_dir,pdf_path.stem)

            all_results[pdf_path.stem] = {
                "table_count": len(tables),
                "tables": [asdict(t) for t in tables]
            }

        total_tables = sum( v["table_count"] for v in all_results.values())
        logger.success(f"Total tables extracted: {total_tables}")

        return all_results