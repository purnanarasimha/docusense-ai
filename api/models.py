"""api pydantic models request and response schemas for all end points"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# request models
class QueryRequest(BaseModel):
    """request body for /query endpind"""

    question : str = Field (
        ...,
        min_length= 3,
        max_length= 1000,
        description= "Question to ask about documents",
        example = "What was total revenue in 2023?"
    )
    top_k : int = Field(
        default= 5,
        ge=1,
        le=20,
        description= "Number of chunks to retrieved"
    )
    doc_id_filter : Optional[str] = Field ( default= None, description= "restrict search to specific document ID")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the attention mechanism?",
                "top_k": 5,
                "doc_id_filter" : None
            }
        }

# response Models

class CitationModel(BaseModel):
    """Single citation"""
    citation_number : int
    doc_id : str
    page_number : int
    content_type : str
    excerpt : str
    relevance_score: float

class CitationBundleModel(BaseModel):
    """All citations for an answer"""
    citations : list[CitationModel]
    total_sources : int
    unique_docs : list[str]
    has_table_source : bool
    has_image_source : bool

class RetrievalStatsModel(BaseModel):
    """Retrieval pipeline statics"""
    semantic_found : int = 0
    bm25_found : int = 0
    after_fusion : int = 0
    after_rerank : int = 0
    query_type : str = "hybrid"
    weights : dict = {}

class QueryResponse(BaseModel):
    """Response from /query endpoint"""

    question : str
    answer : str
    confidence : float
    citations : dict
    retrieval_stats : dict
    doc_ids_used : list[str]
    query_type : str
    has_tables : bool
    chunks_used : int
    processing_time : float
    status : str

    class Config:
        json_schema_extra = {
            "example":{
                "question" : "what is attention?",
                "answer" : "Attendation is a mechanism...",
                "confidence": 0.85,
                "status": "success"
            }
        }

class DocumentInfo(BaseModel):
    """Info about a single ingested document"""
    doc_id : str
    file_name : str
    doc_type : str
    total_pages : int
    total_chunks : int
    file_size_kb : float
    ingested_at : str
    status : str

class DocumentListResponse(BaseModel):
    """Response from GET /documents"""
    documents : list[DocumentInfo]
    total_documents : int
    total_chunks : int

class IngestResponse(BaseModel):
    """Response from POST /ingest"""
    doc_id : str
    file_name : str
    status : str
    total_chunks : int
    text_chunks : int
    table_chunks : int
    image_chunks : int
    processing_time : float
    message : str

class HealthResponse(BaseModel):
    """Response from GET /health"""
    status : str
    version : str
    environment : str
    qdrant_connected : bool
    bm25_loaded : bool
    total_vectors : int

class StatsResponse(BaseModel):
    """Response from GET /stats"""
    total_documents : int
    total_chunks : int
    total_vectors: int
    collections : dict
    bm25_indexed : int
    environment : str