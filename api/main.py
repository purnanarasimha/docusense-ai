"""Fast api appplication main app entry point registers all router and middleware"""

import os
import time
from contextlib import asynccontextmanager
from loguru import logger
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.models import HealthResponse
from api.routers import ingest, query

load_dotenv()

settings = get_settings()

# startup or shutdown

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown Pre-loads heavy components"""

    # start up
    logger.info("="*50)
    logger.info("   Docusense ai start up....")
    logger.info("="*50)

    # pre-load bm25 index
    try:
        from retrieval.bm25_index import BM25Index
        bm25 = BM25Index()
        if not bm25.load():
            logger.warning("BM25 index not found. Run: python -m Retrieval.bm25_index")
        else:
            logger.success("BM25 index loaded")
    except Exception as e:
        logger.warning(f"BM25 pre-load skipped: {e}")

    # test Qdrant connection
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key= os.getenv("QDRANT_API_KEY"), timeout= 10)
        cols = client.get_collections()
        logger.success(f"Qdrant connected: {len(cols.collections)} collections")
    except Exception as e:
        logger.warning(f"Qdrant connection issue: {e}")

    logger.success("DocuSense AI is ready!")
    logger.info(f"Docs at: http://localhost:8000/docs")

    yield

    #shutdown ------
    logger.info("Docusense AI shutting down....")


# app instance

app = FastAPI(title= "DocuSense AI", description="""
MultiModal document intelligence platform Ask Questions about any pdf document using RAG with:
- Hybrid Retrieval (semantic + bm25)
- Multimodal (text + Tables + Images)
- Citations with pages references
- Confidence score per answer

### Quick Start ###
1. upload a pdf via 'POST /ingest'
2. Ask Questions via 'POST /query'
3. Get cited answers with confidence scores""",
version= settings.app_version,
lifespan= lifespan,
docs_url= "/docs",
redoc_url= "/redoc")

# middleware

# cors - allow orgins for demo

app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_credentials= True,
    allow_methods= ["*"],
    allow_headers= ["*"]
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request : Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["x-Process-time"] = (str(round(process_time, 3)))
    return response

# core endpoints
@app.get("/", summary="API Welcome", include_in_schema=False)
async def root():
    return{
        "name" : "DocuSense AI",
        "version" : settings.app_version,
        "status" : "running",
        "docs" : "/docs",
        "health" : "/health",
        "endpoints": {
            "inges": "POST /ingest/",
            "query": "POST /query",
            "documents": "GET /documents",
            "stats": "GET /stats"
        }
    }

@app.get("/health", response_model= HealthResponse, summary="Health check")
async def health_check():
    qdrant_ok = False
    total_vectors = 0
    bm25_loaded = False

    # check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=5)
        collections = client.get_collections()
        qdrant_ok = True

        for col in collections.collections:
            info = client.get_collection(col.name)
            total_vectors += info.points_count

    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")

    # check bm25
    try:
        from retrieval.bm25_index import BM25Index
        from pathlib import Path
        bm25_file = Path("data/processed/bm25_index.pkl")
        bm25_loaded = bm25_file.exists()
    except Exception:
        pass

    return HealthResponse(
        status= "healthy" if qdrant_ok else "degraded",
        version= settings.app_version,
        environment= settings.app_env,
        qdrant_connected= qdrant_ok,
        bm25_loaded= bm25_loaded,
        total_vectors= total_vectors
    )

# Register routers

app.include_router(ingest.router)
app.include_router(query.router)

# exception handlers

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content= {
        "error" : "endpoint not found",
        "docs" : "/docs",
        "path" : str(request.url)
    })

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.error(f"server error: {exc}")
    return JSONResponse ( status_code= 500, content={
        "error": "internal server error",
        "message": str(exc)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0",port=8000, reload=True, log_level="info")