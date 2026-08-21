"""Embedding pipeline converts all chunks into vector embeddings"""

import os
import json
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from loguru import logger
from dotenv import load_dotenv
from tqdm import tqdm

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)

load_dotenv()

# configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# collection names in qdrant
COLLECTION_TEXT     = "docusense_text"
COLLECTION_TABLES   = "docusense_tables"
COLLECTION_IMAGES   = "docusense_images"

# gemini embedding dimension
EMBEDDING_DIM = 3072

BATCH_SIZE = 20
SLEEP_BETWEEN_BATCHES = 2

#paths
BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "processed" / "chunks"
EMBEDDINGS_DIR = BASE_DIR / "data" / "processed" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# embedding generator

class GeminiEmbedder:
    """Generates embeddings using google gemini"""

    def __init__(self):
        self.model = "models/gemini-embedding-001"
        self.max_retries = 3
        self.actual_dim = None

    def get_embedding_dim(self) -> int:
        if self.actual_dim:
            return self.actual_dim

        logger.info("finding gemini embedding dimension")
        test_embe = self.embed_text("text", task_type="retrieval_document")

        if test_embe:
            self.actual_dim = len(test_embe)
            return self.actual_dim

        logger.warning("could not detect dim using 3072")
        self.actual_dim = 3072
        return self.actual_dim

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> Optional[list]:
        """generate embedding for singe text task_type Options retrieval_document - for indexing and retrieval_query for searching"""
        text = text[:8000] if len(text) > 8000 else text

        for attempt in range(self.max_retries):
            try:
                result = genai.embed_content(model=self.model, content=text, task_type=task_type)
                if not self.actual_dim:
                    self.actual_dim = len(result["embedding"])
                return result["embedding"]
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait = (attempt + 1) * 30
                    logger.warning(f"rate limt hit. waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"Embedding error: {e}")
                    if attempt == self.max_retries - 1:
                        return None
                    time.sleep(5)
        return None

    def embed_batch(self, texts: list, task_type: str ="retrieval_document") -> list:
        """ generate embedding for a batch of texts"""

        embeddings = []
        for text in texts:
            emb = self.embed_text(text, task_type)
            embeddings.append(emb)
            time.sleep(0.5)

        return embeddings

# qdrant manager

class QdrantManager:
    """manager Qdrant vector database creates collection, upsert points handles search"""

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL,api_key=QDRANT_API_KEY,timeout=60)
        logger.info("connected to qdrant...")

    def setup_collections(self, embedding_dim: int = 3072):
        """Creating all required collections skip if already exists"""
        collections_config = {
            COLLECTION_TEXT: {
                "size": embedding_dim,
                "description": "Text chunks"
            },
            COLLECTION_TABLES: {
                "size": embedding_dim,
                "description": "Table chunks"
            },
            COLLECTION_IMAGES: {
                "size": embedding_dim,
                "description": "Image caption chunks"
            }
        }

        existing = [c.name for c in self.client.get_collections().collections]

        for name, config in collections_config.items():
            if name in existing:
                logger.info(f"Collection exists: {name}")
                try:
                    col_info = self.client.get_collections(name)
                    stored_dim = col_info.config.params.vectors.size
                    if stored_dim != embedding_dim:
                        logger.warning(f"collection {name} has dim {stored_dim}, need {embedding_dim}. Recreating...")
                        self.client.delete_collection(name)
                        logger.info("deleted old collection")

                    else:
                        logger.info(f"collection is ok : {name} dim={stored_dim}")
                        continue
                except Exception as e:
                    logger.warning(f"could not check dim for {name}: {e}")
                    try:
                        self.client.delete_collection(name)
                    except Exception:
                        pass

            self.client.create_collection(collection_name=name, 
                                          vectors_config=VectorParams(size=config["size"], distance=Distance.COSINE)
                                          )
            logger.success(f"created collection: {name}")

        # create payload indexes for fast filtering
        self._create_indexes()

    def _create_indexes(self):
        """create indexes on metadata fields"""

        for collection in [COLLECTION_TEXT, COLLECTION_IMAGES, COLLECTION_TABLES]:
            try:
                self.client.create_payload_index(collection_name=collection, field_name="doc_id", field_schema=PayloadSchemaType.KEYWORD)
                self.client.create_payload_index(collection_name=collection, field_name="content_type", field_schema=PayloadSchemaType.KEYWORD)
                logger.info(f"indexes created for : {collection}")

            except Exception as e:
                logger.debug(f"Index note for {collection}: {e}")

    def upsert_points(self, collection_name: str, chunks: list, embeddings: list) -> int:
        """Insert or update points in collection Returns count of success ful upserts"""
        points = []
        success = 0

        for chunk, embedding in zip(chunks, embeddings):
            if embedding is None:
                logger.warning(f"Skipping chunk {chunk['chunk_id']} (no embedding)")
                continue

            payload = {
                "chunk_id"      : chunk["chunk_id"],
                "doc_id"        : chunk["doc_id"],
                "content"       : chunk["content"],
                "content_type"  : chunk["content_type"],
                "page_number"   : chunk["page_number"],
                "word_count"    : chunk["word_count"],
                "char_count"    : chunk["char_count"],
                **chunk.get("metadata", {})
            }

            point = PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)

            points.append(point)
            success += 1

        if points:
            for i in range(0, len(points), 100):
                batch = points[i:i + 100]
                self.client.upsert(collection_name=collection_name, points=batch)

        return success

    def get_collection_stats(self) -> dict:
        """get count of points in each collection"""
        stats = {}
        for name in [COLLECTION_IMAGES,COLLECTION_TABLES,COLLECTION_TEXT]:
            try:
                info = self.client.get_collection(name)
                stats[name] = info.points_count
            except Exception:
                stats[name] = 0
        return stats

    def search(self, collection_name: str, query_vector: list, limit: int = 5, doc_id_filter: str = None) -> list:
        """search for similar chunks optional: filter by document"""
        search_filter = None
        if doc_id_filter:
            search_filter = Filter(must=[ FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))])

        results = self.client.search(collection_name=collection_name,
                                     query_vector=query_vector,
                                     limit=limit,
                                     query_filter=search_filter,
                                     with_payload=True)

        return results

# main embedding pipeline

class EmbeddingPipeline:
    """orchestrates embedding generation and storage for all doc chunks"""

    def __init__(self):
        self.embedder = GeminiEmbedder()
        self.qdrant = QdrantManager()

    def run(self) -> dict:
        """embed all chunks and store in qdrant, Processess text, table, image chunks separately"""
        logger.info("="*55)
        logger.info(" embedding pipe line started")
        logger.info("="*55)

        actual_dim = self.embedder.get_embedding_dim()
        logger.info(f"using the embedding dimension: {actual_dim}")
        self.qdrant.setup_collections(embedding_dim=actual_dim)

        # load all chunk files
        chunk_files = list(CHUNKS_DIR.glob("*_chunks.json"))
        if not chunk_files:
            logger.error(f"No chunk files found in {CHUNKS_DIR}")
            return {}

        logger.info(f"found {len(chunk_files)} chunk files")

        text_chunks = []
        table_chunks = []
        image_chunks = []

        for chunk_file in chunk_files:
            with open(chunk_file, encoding="utf-8") as f:
                chunks = json.load(f)

            for chunk in chunks:
                ct = chunk.get("content_type", "text")
                if ct == "text":
                    text_chunks.append(chunk)
                elif ct == "table":
                    table_chunks.append(chunk)
                elif ct == "image_caption":
                    image_chunks.append(chunk)

        logger.info(f"chunks loaded - Text: {len(text_chunks)}, Tables: {len(table_chunks)}, Images: {len(image_chunks)}")

        results = {}

        # embed text chunks
        if text_chunks:
            logger.info("\n Embedding text chunks")
            text_results = self._embed_and_store(chunks=text_chunks, collection_name=COLLECTION_TEXT, label="TEXT")
            results["text"] = text_results

        if table_chunks:
            logger.info("\n embedding table chunks..")
            table_results = self._embed_and_store(chunks=table_chunks, collection_name=COLLECTION_TEXT, label="TABLE")
            results["tables"] = table_results

        if image_chunks:
            logger.info("\n embedding table chunks..")
            image_results = self._embed_and_store(chunks=image_chunks, collection_name=COLLECTION_TEXT, label="IMAGES")
            results["tables"] = image_results

        stats = self.qdrant.get_collection_stats()

        self._print_summary(results, stats)

        return {
            "results"   : results,
            "qdrant_stats": stats
        }

    def _embed_and_store(self, chunks: list, collection_name: str, label: str) -> dict:
        """embed list of chunks and store in qdrant process in batches with progress bar save embeddings locally as bak up"""
        total = len(chunks)
        embedded = 0
        stored = 0
        failed = 0

        #check which chunks already embedded and resumet if interrupted
        embedded_ids = self._load_embedded_ids(label)

        pending = [c for c in chunks if c["chunk_id"] not in embedded_ids]

        if len(pending) < total:
            logger.info(f"Resuming: {total - len(pending)} already done, {len(pending)} are remaining")

        all_embeddings_backup = []

        with tqdm(total=len(pending),desc=f"Embedding {label}",unit="chunk") as pbar:
            for i in range(0, len(pending), BATCH_SIZE):
                batch_chunks = pending[i:i+BATCH_SIZE]

                batch_texts = [c["content"] for c in batch_chunks]
                batch_embeddings = self.embedder.embed_batch(batch_texts,task_type="retrieval_document")

                # store in qdrant
                count = self.qdrant.upsert_points(collection_name=collection_name,chunks=batch_chunks,embeddings=batch_embeddings)

                stored += count
                failed += len(batch_chunks) - count
                embedded += len(batch_chunks)

                # save backup locally
                for chunk, emb in zip(batch_chunks, batch_embeddings):
                    if emb:
                        all_embeddings_backup.append({
                            "chunk_id": chunk["chunk_id"],
                            "doc_id": chunk["doc_id"],
                            "embedding": emb
                        })
                        embedded_ids.add(chunk["chunk_id"])

                # save progress
                self._save_embedded_ids(label,embedded_ids)
                pbar.update(len(batch_chunks))

                if i + BATCH_SIZE < len(pending):
                    time.sleep(SLEEP_BETWEEN_BATCHES)

        backup_path = (EMBEDDINGS_DIR / f"{label.lower()}_embeddings.json")

        with open(backup_path, "w") as f:
            json.dump(all_embeddings_backup, f)

        logger.success(f"{label} complete: {stored} stored, {failed} failed")

        return {
            "total": total,
            "stored": stored,
            "failed": failed
        }

    def _load_embedded_ids(self, label: str) -> set:
        """load set of already embedded chunk ids"""
        progress_file = (
            EMBEDDINGS_DIR/ f"{label.lower()}_progress.json"
        )
        if progress_file.exists():
            with open(progress_file) as f:
                return set(json.load(f))
        return set()

    def _save_embedded_ids(self, label: str, ids: set):
        """save progress to resume if interrupted"""
        progress_file = (
                    EMBEDDINGS_DIR/ f"{label.lower()}_progress.json"
                )
        if progress_file.exists():
            with open(progress_file, "w") as f:
                json.dump(list(ids), f)

    def _print_summary(self, results: dict, stats: dict):
        print("\n" + "="*60)
        print(" EMBEDDING PIPELINE COMPLETE")
        print("="*60)

        for content_type,  result in results.items():
            print(f"\n {content_type.upper()}")
            print(f"total chunks: {result['total']}")
            print(f"stored: {result['stored']}")
            print(f"failed: {result['failed']}")

        print("\n Qdrant collections")
        for collection, count in stats.items():
            short = collection.replace("docusense_", "")
            print(f" {short:<12} : {count} vectors")

        total_vectors = sum(stats.values())
        print("\n" + "-"* 60)
        print(f" total vectors in Qdrant: {total_vectors}")
        print("="*60)
        print("ready for Retrieval system")
        print("="*60)

# quick search test
def test_search(query: str = "what is attention mechanism?"):
    """simple test"""
    logger.info(f"\n Test search: '{query}'")

    embedder = GeminiEmbedder()
    qdrant = QdrantManager()

    query_vector = embedder.embed_text(query,task_type="retrieval_query")

    if not query_vector:
        logger.error("failed to embed the query")
        return

    print("\n"+"="*60)
    print(f"search test for '{query}'")
    print("="*60)

    for collection in [COLLECTION_TEXT,COLLECTION_TABLES]:
        results = qdrant.search(collection_name=collection, query_vector=query_vector,limit=2)

        short = collection.replace("docusense_", "").upper()
        print(f"\n [{short}] Top {len(results)} results:")

        for i, result in enumerate(results,1):
            print(f"\n Result {i}:")
            print(f"score: {result.score:.4f}")
            print(f"doc:{result.payload.get('doc_id','?')}")
            print(f"page:{result.payload.get('page_number','?')}")
            content = result.payload.get('content' '')[:150]
            print(f" content: {content}...")

    print("\n" + "="*55)
    print("search working")
    print("="*55)

# run
if __name__ == "__main__":
    pipeline = EmbeddingPipeline()
    result = pipeline.run()

    print("running simple search test...")
    time.sleep(2)
    test_search("what is attendanction mechanish in transformers?")
    test_search("what was total revenue?")