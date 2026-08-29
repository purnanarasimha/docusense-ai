"""context Assembler takes retrieved chunks and assembles them into optimized llm conterxt"""

import json
from pathlib import Path
from dataclasses import dataclass
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# config

MAX_CONTEXT_CHARS = 12000
MAX_CHUNKS_IN_CTX = 8

# data models
class AssembledContext:
    """Final context ready for LLM"""
    context_text: str
    source_chunks: list
    total_chars: int
    chunk_count: int
    doc_ids_used: list
    has_tables: bool
    has_images: bool
    truncated: bool

# Context assembler

class ContextAssembler:
    """Assembles retrieved chunks into structured context for the LLM"""
    def __init__(self, max_chars: int = MAX_CONTEXT_CHARS, max_chunks: int= MAX_CHUNKS_IN_CTX):
        self.max_chars = max_chars
        self.max_chunks = max_chunks

    def assemble(self, chunks: list, query: str, doc_filter: str = None) -> AssembledContext:
        """main assebly method, takes chunks returns formatted context"""
        if not chunks:
            logger.warning("No chunks are provided for the context assembly")
            return self._empty_context()

        # filter by doc if requested
        if doc_filter:
            chunks = [c for c in chunks if c.get("doc_id")==doc_filter]

        # Deduplicate
        chunks = self._deduplicate(chunks)

        # sort - first tables then text
        chunks = self._sort_chunks(chunks)

        # limit to max chunks
        chunks = chunks[:self.max_chunks]

        # build context
        context_parts = []
        source_chunks = []
        total_chars = 0
        truncated = False

        has_tables = False
        has_images = False

        for idx, chunk in enumerate(chunks):
            content_type = chunk.get("content_type", "text")
            content = chunk.get("content", "")
            doc_id = chunk.get("doc_id", "unknown")
            page_num = chunk.get("page_number", "?")

            # format based on content type
            formatted = self._format_chunk(
                chunk_idx = idx + 1,
                content = content,
                content_type = content_type,
                doc_id = doc_id,
                page_num = page_num
            )

            # check context windown
            if total_chars + len(formatted) > self.max_chars:
                logger.warning(f"Context limit reached at chunk {idx + 1}")
                truncated = True
                break

            context_parts.append(formatted)
            source_chunks.append(chunk)
            total_chars += len(formatted)

            # track content types
            if content_type == "table":
                has_tables = True
            elif content_type == "image_caption":
                has_images = True

        # compbine all parts
        context_text = "\n\n".join(context_parts)

        # get unique doc ids used
        doc_ids_used = list(set(c.get("doc_id", "unknown") for c in source_chunks))

        assembled = AssembledContext(
            context_text = context_text,
            source_chunks = source_chunks,
            total_chars = total_chars,
            chunk_count = len(source_chunks),
            doc_ids_used = doc_ids_used,
            has_tables = has_tables,
            has_images = has_images,
            truncated = truncated
        )

        logger.info(f"context assembled: {assembled.chunk_count} chunks, {assembled.total_chars} chars, docs={assembled.doc_ids_used}")
        return assembled

    def _format_chunk(self,chunk_idx: int, content: str, content_type: str, doc_id: str, page_num: int) -> str:
        """Format chunk based on its content type clear source markers help LLM cire correctly"""
        source_label = f"[Source {chunk_idx}: {doc_id}, Page {page_num}]"

        if content_type == "table":
            return(f"{source_label} [TABLE DATA]\n {'-' * 40}\n {content}\n {'-' * 40}")

        elif content_type == "image_caption":
            return(f"{source_label} [IMAGE DESCRIPTION]\n {content}")
        else:
            return(f"{source_label}\n {content}")

    def _deduplicate(self, chunks: list) -> list:
        """remove duplicate chunks by chunk id"""
        seen = set()
        unique = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk)

        removed = len(chunks) - len(unique)
        if removed > 0:
            logger.info(f"removed {removed} duplicate chunks")

        return unique

    def _sort_chunks(self, chunks: list) -> list:
        """sort chunks for optiomal context ordering: table chunks first (structured data) then text by relevance"""
        def sort_key(chunk):
            content_type = chunk.get("content_type", "text")
            rrf_score = chunk.get("rrf_score", 0)

            # tables get priority boost
            type_priority ={"table":2, "image_caption":1, "text":0}
            priority = type_priority.get(content_type, 0)

            return (priority, rrf_score)
        return sorted(chunks, key=sort_key, reverse=True)

    def _empty_context(self) -> AssembledContext:
        """Return empty context when no chunks available"""
        return AssembledContext(
            context_text = "",
            source_chunks = [],
            total_chars = 0,
            chunk_count = 0,
            doc_ids_used = [],
            has_tables = False,
            has_images = False,
            truncated = False
        )