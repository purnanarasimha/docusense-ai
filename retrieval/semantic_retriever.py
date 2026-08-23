"""semantic retriever search qdrant vector collection using gemini embeddings"""
import os
import time
from loguru import logger
from dotenv import load_dotenv
from typing import Optional

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

# config
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_TEXT = "docusense_text"
COLLECTION_TABLES = "docusense_tables"
COLLECTION_IMAGES = "docusense_images"

# semantic retriever

class SemanticRetriever:
    """ Retrieves relevant chunk using vector similiraty search in Qdrant handle text table images"""

    def __init__(self):
        self.qdrant = QdrantClient(url=QDRANT_URL,api_key=QDRANT_API_KEY,timeout=30)
        self.embed_model = "models/gemini-embeddding-001"
        self.max_retries = 3

    # embed Query

    def embed_query(self, query: str) -> Optional[list]:
        """ embed user query for similarity search user retieval query as task type"""

        query = query[:8000]

        for attempt in range(self.max_retries):
            try:
                result = genai.embed_content(model=self.embed_model,content=query,task_type="retrieval_query")
                return result["embedding"]

            except Exception as e:
                if "429" in str(e):
                    wait = (attempt + 1) * 15
                    logger.warning(f"rate limit, waiting {wait}s..")
                    time.sleep(wait)
                else:
                    logger.error(f"Embed query error: {e}")
                    if attempt == self.max_retries -1:
                        return None
                    time.sleep(3)
        return None

    def search_text(self, query_vector: list, top_k: int = 10, doc_id_filter: str = None, score_threshold: float = 0.3) -> list:
        """search text collection"""
        return self._search_collection(collection=COLLECTION_TEXT,query_vector=query_vector,top_k=top_k,doc_id_filter=doc_id_filter,score_threshold=score_threshold,retriever_label="semantic_text")

    def search_tables(self, query_vector: list, top_k: int = 5, doc_id_filter: str = None, score_threshold: float = 0.3) -> list:
        """search table collection"""
        return self._search_collection(collection=COLLECTION_TABLES,query_vector=query_vector,top_k=top_k,doc_id_filter=doc_id_filter,score_threshold=score_threshold,retriever_label="semantic_table")

    def search_images(self, query_vector: list, top_k: int = 3, doc_id_filter: str = None, score_threshold: float = 0.3) -> list:
        """search image caption collection"""
        return self._search_collection(collection=COLLECTION_IMAGES,query_vector=query_vector,top_k=top_k,doc_id_filter=doc_id_filter,score_threshold=score_threshold,retriever_label="semantic_image")

    def search_all(self, query: str, top_k: int = 10, doc_id_filter: str = None) -> dict:
        """Search All collections with once query to retun results from each collection"""

        #embed query once and reues for all seraches
        query_vector = self,self.embed_query(query)

        if not query_vector:
            logger.error("failec to embed the query")
            return {"text": [], "tables": [], "images": []}

        text_results = self.search_text(query_vector,top_k,doc_id_filter)
        table_results = self.search_tables(query_vector, top_k // 2, doc_id_filter)
        image_results = self.search_images(query_vector, 3, doc_id_filter)

        return {
            "text": text_results,
            "tables": table_results,
            "images": image_results,
            "query_vector": query_vector
        }

    def _search_collection(self, collection: str, query_vector: list, top_k: int, doc_id_filter: str, score_threshold: float, retriever_label: str) -> list:
        """search a specific qdrant collection"""
        try: 
            search_filter = None
            if doc_id_filter:
                search_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))])

            hits = self.qdrant.search(collection_name=collection,query_vector=query_vector,limit=top_k,query_filter=search_filter,with_payload=True, score_threshold=score_threshold)

            results = []
            for hit in hits:
                results.append({
                    "chunk_id": hit.payload.get("chunk_id"),
                    "doc_id": hit.payload.get("doc_id"),
                    "content": hit.payload.get("content"),
                    "content_type": hit.payload.get("content_type"),
                    "page_number": hit.payload.get("page_number"),
                    "semantic_score": hit.score,
                    "retriever": retriever_label,
                    "metadata": {
                        k: v for k, v in hit.payload.items()
                        if k not in [
                            "chunk_id", "doc_id", "content", "content_type","page_number"
                        ]
                    }
                })

            return results
        except Exception as e:
            logger.error(f"search failed in {collection}: {e}")
            return []