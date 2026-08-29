"""Cititaion Builder to build structured cititations from source chunks"""

from dataclasses import dataclass, asdict
from loguru import logger

# data models
@dataclass
class Citation:
    citation_number: int
    doc_id: str
    page_number: int
    content_type: str
    excerpt: str
    relevance_score: float

@dataclass
class CitationBundle:
    citations: list
    total_sources: int
    unique_docs: list
    has_table_source: bool
    has_image_source: bool

# cititation builder
class CitationBuilder:
    """Build citation from source chunks to make answers verifiable and trustworthy"""
    def __init__(self, excerpt_length: int = 150):
        self.excerpt_length = excerpt_length

    def build(self, source_chunks: list) -> CitationBundle:
        """build citation bundle from shource chunks args source_chunks returns citation bundle with all citations"""
        if not source_chunks:
            return self._empty_bundle()

        citations = []
        unique_docs = list(set(c.get("doc_id", "unkown") for c in source_chunks))
        has_table = any(c.get("content_type") == "table" for c in source_chunks)
        has_image = any(c.get("content_type") == "table" for c in source_chunks)

        for idx, chunk in enumerate(source_chunks):
            # get excerpt (forst N chars of content)
            content = chunk.get("content", "")
            excerpt = content[:self.excerpt_length]
            if len(content) > self.excerpt_length:
                excerpt += "..."

            citation = Citation(
                citation_number=idx + 1,
                doc_id= chunk.get("doc_id", "unknown"),
                page_number= chunk.get("page_number", 0),
                content_type=chunk.get("content_type", "text"),
                excerpt= excerpt,
                relevance_score= round(chunk.get("rrf_score", 0), 4)
            )
            citations.append(citation)

        bundle = CitationBundle(
            citations = [asdict(c) for c in citations],
            total_sources= len(citations),
            unique_docs= unique_docs,
            has_table_source= has_table,
            has_image_source= has_image
        )
        logger.info(f"Build {len(citations)} citations from {len(unique_docs)} documents")

        return bundle

    def format_for_display(self, bundle: CitationBundle) -> str:
        """Format citations as readable text append to end of answer"""
        if not bundle.citations:
            return ""
        lines = ["\n\n --- \n**sources:**"]

        for cite in bundle.citations:
            doc = cite["doc_id"].replace("_", " ").title()
            page = cite["page_number"]
            ctype = cite["content_type"].replace("_", " ").title()

            lines.append(f"[{cite['citation_number']}] **{doc}** - Page {page} ({ctype})")

        return "\n".join(lines)

    def _empty_bundle(self) -> CitationBundle:
        return CitationBundle(
            citations= [],
            total_sources= 0,
            unique_docs= [],
            has_table_source= False,
            has_image_source= False
        )

