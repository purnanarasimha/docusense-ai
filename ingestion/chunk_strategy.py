""" Smart chuncking strategy - to convert parsed documents into optimized chunks"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# data models

@dataclass
class DocumentChunk:
    """single chunk ready for embedding"""
    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    content_type: str
    page_number: int
    char_count: int
    word_count: int
    metadata: dict

# chunker class
class SmartChunker:
    """Creates optimized chunks from parsed documents 
    text - sentance aware sliding window
    tables - one chunk per table image - one chunk per images"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, parsed_path: Path, tables_path: Path = None, images_path: Path = None) -> list:
        """create all chunks for document and combine text table and image"""
        all_chunks = []

        # text chunks from parsed content
        if parsed_path.exists():
            text_chunks = self._chunk_text_content(parsed_path)
            all_chunks.extend(text_chunks)
            logger.info(f"Text chunks created: {len(text_chunks)}")

        # 2. table chunks
        if tables_path and tables_path.exists():
            table_chunks = self._chunk_tables(tables_path)
            all_chunks.extend(table_chunks)
            logger.info(f"table chunks created: {len(table_chunks)}")

        if images_path and images_path.exists():
            image_chunks = self._chunk_image_captions(images_path)
            all_chunks.extend(image_chunks)
            logger.info(f"Image chunks created: {len(image_chunks)}")

        logger.success(f"Total chunks for document: {len(all_chunks)}")

        return all_chunks

    def _chunk_text_content(self, parsed_path: Path) -> list:
        """create chunks from parsed text content"""

        chunks = []
        with open(parsed_path, encoding="utf-8") as f:
            parsed_doc = json.load(f)

        doc_id = parsed_doc["doc_id"]
        doc_type = parsed_doc["doc_type"]
        chunk_index = 0

        for page_data in parsed_doc["pages"]:
            page_num = page_data["page_number"]
            page_text = page_data["text"]

            if not page_text or len(page_text.strip()) < 20:
                continue

            page_chunks = self._split_into_chunks(page_text)

            for chunk_text in page_chunks:
                if len(chunk_text.strip()) < 20:
                    continue

                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_text_{chunk_index}",
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    content=chunk_text.strip(),
                    content_type="text",
                    page_number=page_num,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    metadata={
                        "doc_type": doc_type,
                        "source": parsed_doc["file_name"],
                        "page": page_num,
                        "total_pages": parsed_doc["total_pages"],
                        "has_images": page_data.get("has_images", False),
                        "has_tables": page_data.get("has_tables", False),
                    }
                )

                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _split_into_chunks(self, text:str) -> list:
        """split text into overlapping chunks"""

        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if sentence_size > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                word_chunks = self._split_by_words(sentence)
                chunks.extend(word_chunks)
                continue

            if current_size + sentence_size > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_sentences(self, text: str) -> list:
        """split text into sentences"""
        # simple sentence splitter
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_by_words(self, text: str) -> list:
        """split long text by word count"""
        words = text.split()
        chunks = []
        size = self.chunk_size // 5

        for i in range(0, len(words), size):
            chunk = " ".join(words[i:i+size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def _chunk_tables(self, tables_path: Path) -> list:
        """create one chunk per table"""
        chunks = []

        with open(tables_path, encoding="utf-8") as f:
            tables = json.load(f)

        for table in tables:
            content = (f"TABLE DATA:\n {table.get('raw_text', '')}\n\n MARKDOWN FORMAT:\n {table.get('markdown_text', '')}")
            chunk = DocumentChunk(
                chunk_id=table["table_id"],
                doc_id=table["doc_id"],
                chunk_index=0,
                content=content,
                content_type="table",
                page_number=table["page_number"],
                char_count=len(content),
                word_count=len(content.split()),
                metadata={
                    "table_id": table["table_id"],
                    "row_count": table["row_count"],
                    "col_count": table["col_count"],
                    "headers": table["headers"],
                    "source": table["doc_id"],
                    "page": table["page_number"],
                }
            )
            chunks.append(chunk)

        return chunks

    def _chunk_image_captions(self, images_path: Path) -> list:
        """create one chunk per image captions"""
        chunks = []

        with open(images_path, encoding="utf-8") as f:
            images = json.load(f)

        for image in images:
            if not image.get("caption"):
                continue

            content = (f"IMAGE DESCRIPTION (Page {image['page_number']}):\n {image['caption']}")

            chunk = DocumentChunk(
                chunk_id=image["image_id"],
                doc_id=image["doc_id"],
                chunk_index=0,
                content=content,
                content_type="image_caption",
                page_number=image["page_number"],
                char_count=len(content),
                word_count=len(content.split()),
                metadata={
                    "image_id": image["image_id"],
                    "image_width": image["width"],
                    "image_heighe": image["height"],
                    "source": image["doc_id"],
                    "page": image["page_number"],
                }
            )

            chunks.append(chunk)

        return chunks

    def save_chunks(self, chunks: list, output_dir: Path, doc_id: str) -> Path:
        """save chunks to Json"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{doc_id}_chunks.json"

        chunks_data = [asdict(c) for c in chunks]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        logger.info(f"saved {len(chunks)} chunks to {output_path.name}")
        return output_path