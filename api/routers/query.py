"""Query router handles question answering endpints
POST /query -> ask questiong, get answer
GET /documents -> list all documents
GET /stats -> system Statistics"""

import os
import json
from pathlib import Path
from loguru import logger
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

from api.models import QueryRequest,QueryResponse,DocumentListResponse,DocumentInfo,StatsResponse

load_dotenv()

#Paths
BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
METADATA_FILE = BASE_DIR / "data" / "document_metadata.json"

router = APIRouter(tags=["Query"])

# global enging (loaded once)

_engine = None

def get_engine():
    """Lazy load the docusense engine sigleton pattern - only init once"""

    global _engine
    if _engine is None:
        logger.info("Loading DocuSense Engine...")
        from generation.docusense_engine import DocuSenseEngine
        _engine = DocuSenseEngine(use_gemini_reranker=False)
        logger.success("Engine loaded!")
    return _engine

# Routes 
@router.post("/query", response_model=QueryResponse, summary="Ask a question about your documents")
async def query_documents(request: QueryRequest):
    """Ask any question about ingested documents the system will return with citations and confidense score"""
    try:
        engine = get_engine()
        response = engine.query(
            question= request.question,
            top_k= request.top_k,
            doc_id_filter= request.doc_id_filter
        )

        return QueryResponse(
            question= response.question,
            answer= response.answer,
            confidence= response.confidence,
            citations= response.citations,
            retrieval_stats= response.retrieval_stats,
            doc_ids_used= response.doc_ids_used,
            query_type= response.query_type,
            has_tables= response.has_tables,
            chunks_used= response.chunks_used,
            processing_time= response.processing_time,
            status= response.status
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code= 500, detail= f"Query Process failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse, summary="list all ingested documents")
async def list_documents():
    """Returns list of all ingested documents with their metadata and chunk counts"""
    try:
        # load metadata
        metadata ={}
        if METADATA_FILE.exists():
            with open(METADATA_FILE, encoding="utf-8") as f:
                metadata = json.load(f)

        documents = []
        total_chunks = 0

        for doc_id, meta in metadata.items():
            # get chunk count from chunk files.
            chunk_file = PROCESSED_DIR / "chunks" / f"{doc_id}_chunks.json"
            chunk_count = 0
            if chunk_file.exists():
                with open(chunk_file, encoding="utf-8") as f:
                    chunks = json.load(f)
                    chunk_count = len(chunks)

            total_chunks += chunk_count

            # get page count from parsed file
            parsed_file = PROCESSED_DIR / "parsed" / f"{doc_id}_parsed.json"
            page_count = 0
            if parsed_file.exists():
                with open(parsed_file, encoding="utf-8") as f:
                    parsed = json.load(f)
                    page_count = parsed.get("total_pages", 0)

            documents.append(DocumentInfo(
                doc_id= doc_id,
                file_name= meta.get("file_name", f"{doc_id}.pdf"),
                doc_type= meta.get("doc_type", "general"),
                total_pages= page_count,
                total_chunks= chunk_count,
                file_size_kb= meta.get("file_size_kb", 0),
                ingested_at= meta.get("status", "unknown")
            ))

        return DocumentListResponse(
            documents= documents,
            total_documents= len(documents),
            total_chunks= total_chunks
        )

    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code= 500, detail= str(e))

@router.get("/stats", response_model=StatsResponse, summary="get system statistics")
async def get_stats():
    try:
        #metada stats
        metadata = {}
        if METADATA_FILE.exists():
            with open(METADATA_FILE, encoding="utf-8") as f:
                metadata = json.load(f)

        # count total chunks
        total_chunks = 0
        chunks_dir = PROCESSED_DIR / "chunks"
        if chunks_dir.exists():
            for cf in chunks_dir.glob("*_chunks.json"):
                with open(cf, encoding="utf-8") as f:
                    total_chunks += len(json.load(f))
        #qdrant stats
        collections = {}
        total_vectors = 0
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                url= os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                Timeout = 10
            )
            for name in ["docusense_text", "docusense_tables", "docusense_images"]:
                try:
                    info = client.get_collection(name)
                    count = info.points_count
                    collections[name] = count
                    total_vectors +=count
                except Exception:
                    collections[name] = 0
        except Exception as e:
            logger.warning(f"Qdrant stats failed: {e}")

        # bm25 stats
        bm25_indexed = 0
        bm25_corpus = PROCESSED_DIR / "bm25_index" / "bm25_corpus.json"
        if bm25_corpus.exists():
            with open(bm25_corpus, encoding="utf-8") as f:
                bm25_indexed = len(json.load(f))

        return StatsResponse(
            total_documents= len(metadata),
            total_chunks= total_chunks,
            total_vectors= total_vectors,
            collections= collections,
            bm25_indexed= bm25_indexed,
            environment= os.getenv("APP_ENV", "development")
        )

    except Exception as e:
        logger.error(f"stats filed: {e}")
        raise HTTPException(status_code=500, detail=str(e))