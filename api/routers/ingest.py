""" Ingest router hangles document upload and processing
POST /ingest -->  upload pdf, process, embed
Delete /Documents/{doc_id} -> remove document"""

import os
import json
import shutil
import time
from pathlib import Path
from loguru import logger

from fastapi import (APIRouter,UploadFile,File,HTTPException,BackgroundTasks)

from api.models import IngestResponse
from api.config import get_settings

settings = get_settings()

# paths

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
METADATA_FILE = BASE_DIR / "data" / "document_metadata.json"

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# helper functions

def load_metadata() -> dict:
    """load document medata registery"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding="utf-8") as f:
            return json.load(f)

    return {}

def save_metadata(metadata: dict):
    """Save document metadata registry"""
    METADATA_FILE.parent.mkdir(parents=True, exist_ok= True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def detect_doc_type(filename: str) -> str:
    """Detect document type from filename"""
    name = filename.lower()
    if any(k in name for k in ["invoice", "inv_"]):
        return "invoice"
    elif any(k in name for k in ["financial", "income", "10k"]):
        return "financial_report"
    elif any(k in name for k in ["contract", "agreement"]):
        return "contract"
    elif any(k in name for k in ["paper", "research", "arxiv"]):
        return "research_paper"
    return "general"

# routes

@router.post("/", response_model=IngestResponse,summary="upload and ingest a pdf document")
async def ingest_document(background_tasks: BackgroundTasks, file : UploadFile = File(...)):
    """Upload pdf document for ingestion
    process: 
        1.validate file is pdf or not
        2. save to raw directory
        3. parse text + tables
        4. create chunks
        5. generated embeddings
        6. store in qdrant"""

    start_time = time.time()

    # -- validate file type -----------
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400,
                            detail= "Only PDF files are supported")

    # -- check file size 50 mb limt
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > 50:
        raise HTTPException(status_code=413, detail=f"File is too large: {size_mb:.1f}MB. Max allowed is 50MB.")

    # save PDF
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = RAW_DIR / file.filename
    doc_id = pdf_path.stem

    try:
        with open(pdf_path, "wb") as f:
            f.write(content)
        logger.info(f"saved pdf: {file.filename} ({size_mb:.1f}MB)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # -- run ingestion pipeline
    try: 
        from ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        chunks = pipeline.run_single(pdf_path=pdf_path, skip_images=True)

        # -- embed chunks -----------
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embed_pipeline = EmbeddingPipeline()

        # save chunks to file first
        chunks_path = (PROCESSED_DIR / "chunks" / f"{doc_id}_chunks.json")

        if chunks_path.exists():
            embed_pipeline.run()

        # count chunks by type
        text_count = sum(1 for c in chunks if (c if isinstance(c, dict) else vars(c)).get("content_type") == "text")
        table_count = sum(1 for c in chunks if (c if isinstance(c, dict) else vars(c)).get("content_type") == "table")
        image_count = sum(1 for c in chunks if (c if isinstance(c, dict) else vars(c)).get("content_type") == "image_caption")

        # Rebuild BM25 index
        background_tasks.add_task(rebuild_bm25_index)

        # save metadata
        metadata = load_metadata()
        metadata[doc_id] = {
            "doc_id" : doc_id,
            "file_name" : file.filename,
            "doc_type" : detect_doc_type(file.filename),
            "total_chunks" : len(chunks),
            "file_size_kb" : round(size_mb * 1024, 1),
            "status" : "ingested"
        }
        save_metadata(metadata)
        processing_time = time.time() - start_time

        return IngestResponse(
            doc_id= doc_id,
            file_name= file.filename,
            status= "success",
            total_chunks= len(chunks),
            text_chunks= text_count,
            table_chunks= table_count,
            image_chunks= image_count,
            processing_time= round(processing_time, 2),
            message= f"Successfully ingested {file.filename}. Created {len(chunks)} searchable chunks"
        )

    except Exception as e: 
        logger.enable(f"ingestion failed: {e}")
        # clean up on failure
        if pdf_path.exists():
            pdf_path.unlink()
        raise HTTPException(status_code=500, detail= f"Ingestion failed: {str(e)}")

@router.delete("/{doc_id}", summary="remove a document from the system")
async def delete_document(doc_id: str):
    """remove a document and all its data"""

    removed = []
    errors = []

    # remove pdf
    pdf_path = RAW_DIR / f"{doc_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
        removed.append("pdf")

    # remove processed files
    for subdir in ["parsed", "tables", "chunks", "images"]:
        for pattern in [ f"{doc_id}_*.json", f"{doc_id}/"]:
            for f in (PROCESSED_DIR / subdir).glob(pattern):
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
                removed.append(f"{subdir}/{f.name}")

    # remove from metadata
    metadata = load_metadata()
    if doc_id in metadata:
        del metadata[doc_id]
        save_metadata(metadata)
        removed.append("metadata")

    # remove from qdrant
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        import os

        client = QdrantClient(url= os.getenv("QDRANT_URL"), api_key= os.getenv("QDRANT_API_KEY"))

        for collection in ["docusense_text", "docusense_tables", "docusense_images"]:
            client.delete(collection_name= collection, points_selector= Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]))
        removed.append("qdrant_vectors")

    except Exception as e:
        errors.append(f"Qdrant cleanup: {str(e)}")

    return {
        "doc_id" : doc_id,
        "status" : "deleted",
        "removed" : removed,
        "errors" : errors
    }

# background tasks

def rebuild_bm25_index():
    """Rebuild bm25 index after new document"""
    try:
        from retrieval.bm25_index import BM25Index
        index = BM25Index()
        count = index.build()
        logger.info(f"BM25 index rebuilt: {count} chunks")
    except Exception as e:
        logger.error(f"BM25 rebuild failed: {e}")