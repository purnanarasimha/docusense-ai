""" Hybrid Retriever main retrieval orchestartor: route query to stategy
 semantic search, bm25 keyword search, rrf fusion, rerank final results"""

import os
from loguru import logger
from dotenv import load_dotenv

from retrieval.semantic_retriever import SemanticRetriever
from retrieval.bm25_index import BM25Index
from retrieval.rrf_fusion import RRFFusion, QueryRouter
from retrieval.reranker import SimpleReranker, GeminiReranker

load_dotenv()

class HybridRetriever:
    """combines semantic + keyword search with intelligent routing and reranking"""

    def __init__(self, use_gemini_reranker: bool = False):
        """use_gemini_reranker: True - give better quality, keepting false to save api calls"""
        logger.info("Initializing hybrid retriever...")

        self.semantic = SemanticRetriever()
        self.bm25 = BM25Index()
        self.fusion = RRFFusion(k=60)
        self.router = QueryRouter()

        if use_gemini_reranker:
            self.reranker = GeminiReranker()
            logger.info("using gemini reranker for better quality")
        else:
            self.reranker = SimpleReranker()
            logger.info("using simple reranker for faster response")

        # load bm25 index
        self.bm25_loaded = self.bm25.load()
        if not self.bm25_loaded:
            logger.warning("BM25 index not fount run bm25 index build() first")

    def retrieve(self, query: str, top_k: int = 5, doc_id_filter: str = None, include_tables: bool = True) -> dict:
        """main Retrieval method Returns top_k most relevant chunks"""
        logger.info(f"Retrieving for : '{query[:60]}...'")

        query_type = self.router.classify_query(query)
        weights = self.router.get_weights(query_type)

        logger.info(f"Query type: {query_type} | weights: semantic={weights['semantic']}, bm25={weights['bm25']}")

        # semantic search
        semantic_results = self.semantic.search_all(query=query, top_k=top_k*2, doc_id_filter=doc_id_filter)

        query_vector = semantic_results.pop("query_vector", None)
        text_results = semantic_results.get("text", [])
        table_results = semantic_results.get("tables", [])

        # combine semantic text + table results
        all_semantic = text_results
        if include_tables:
            all_semantic = text_results + table_results

        logger.info(f"semantic: {len(text_results)} text, {len(table_results)} table results")

        # bm25 key work search
        bm25_results = []
        if self.bm25_loaded:
            bm25_results = self.bm25.search(query=query,top_k=top_k*2,doc_id_filter=doc_id_filter)
            logger.info(f"bm25: {len(bm25_results)} results")
        else:
            logger.warning("bm25 is not available using semantic only")

        # rrf fusion
        fused_results = self.fusion.fuse(result_lists=[all_semantic, bm25_results],
                                         weights=[weights["semantic"],weights["bm25"]])
        logger.info(f"after fusion: {len(fused_results)} unique chunks")

        # rerank
        final_results = self.reranker.rerank(query=query, chunks=fused_results,top_k=top_k)
        logger.success(f"final: {len(final_results)} chunks retrieved")

        # build
        return {
            "chunks": final_results,
            "query_type": query_type,
            "retrieval_stats": {
                "semantic_found": len(all_semantic),
                "bm25_found": len(bm25_results),
                "after_fusion": len(fused_results),
                "after_rerank": len(final_results),
                "query_type": query_type,
                "weights": weights
            }
        }

    def retrieve_for_display(self, query: str, top_k: int = 5) -> None:
        """printing retrieval results for testing and debugging"""
        result = self.retrieve(query, top_k)

        print("\n"+ "="*60)
        print(f"Query: {query}")
        print(f"Type: {result['query_type']}")
        print("=" * 60)

        stats = result["retrieval_stats"]
        print(f"\n stats:")
        print(f"    semantic found: {stats['semantic_found']}")
        print(f"    bm25 found: {stats['bm25_found']}")
        print(f"    after fusion: {stats['after_fusion']}")
        print(f"    final retured: {stats['after_rerank']}")

        print(f"\n Top {len(result['chunks'])} results:")
        for i , chunk in enumerate(result["chunks"], 1):
            print(f"\n [{i}] {chunk.get('doc_id', '?')}")
            print(f"    Type: {chunk.get('content_type','?')}")
            print(f"    Page: {chunk.get('page_number','?')}")
            print(f"    RRF: {chunk.get('rrf_score',0):.4f}")
            print(f"    Found by: {chunk.get('found_by',[])}")

            content = chunk.get("conternt", "")[:150]
            print(f"    Content: {content}...")
        print("\n" + "="* 60)