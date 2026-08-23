"""Reranker re-scores fused results using femini for final ordering
emilinate all low quality chunks before sending to llm"""

import os
import time
from loguru import logger
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiReranker:
    """Uses gemini to rerank retrieved chunks by relevance to the query"""

    def __init__(self):
        self.model = genai.GenerativeModel("gemini-3.5-flash-lite")
        self.max_to_rerank = 10
        self.min_score = 3

    def rerank(self, query: str, chunks: list, top_k: int = 5) -> list:
        """rerank chunks by relevance to query
        Returns top_k most relevant chunks"""
        if not chunks:
            return []

        # only rerank top ones
        candidates = chunks[:self.max_to_rerank]
        logger.info(f"reranking only {len(candidates)} for the give query: '{query[:50]}..'")
        scored = []
        for chunk in candidates:
            score = self._score_chunk(query=query, chunk_content=chunk.get("content", ""), content_type=chunk.get("content_type", "text"))

            if score >= self.min_score:
                chunk_copy = chunk.copy()
                chunk_copy["rerank_score"] = score
                scored.append(chunk_copy)

            time.sleep(0.3)

        # sort by rerank score
        scored.sort(key=lambda x: x.get("rerank_score", 0), reverse= True)

        # fall back to rrf order if re ranking filtered too much
        if len(scored) < 2:
            logger.warning("reranking fileter too many chunks falling back to rrf order")
            for chunk in candidates:
                chunk_copy = chunk.copy()
                chunk_copy["rerank_score"] = 0
            return candidates[:top_k]

        result = scored[:top_k]
        logger.success(f"reranked: {len(result)} chunks selected")

        return result

    def _score_chunk(self, query: str, chunk_content: str, content_type: str) -> int:
        """score a single chunks relevanct and return score in 1 to 5"""
        content_preview = chunk_content[:500]

        prompt = f"""Rate how relevant this document chunk is to answering the users question.
        Question: {query}
        Document chunks ({content_type}): {content_preview}
        Rate the relevance on the scaler of 1 to 5
        5 - Directly answers the question
        4 - highly relevant, contains key information
        3 - relevant, has partial information
        2 - loosely related
        1 - not relevant
        Strictly respond with only a single digit (1,2,3,4,5). don't provide any explanation"""

        try:
            response = self.model.generate_content(prompt)
            score_text = response.text.strip()

            # extract digit
            for char in score_text:
                if char.isdigit() and char in "12345":
                    return int(char)

            return 3

        except Exception as e:
            logger.warning(f"rerank scoreing failed: {e}")
            return 3

class SimpleReranker:
    """lightweight re ranker using keyword overlap 
    No api calls needed - instant use thi if gemini reranker is too slow"""

    def rerank(self, query: str, chunks: list, top_k: int = 5) -> list:
        """Rerank by keywork overlap score"""
        query_terms = set(query.lower().split())

        scored = []
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            content_terms = set(content.split())

            # keyword overlap score
            overlap = len(query_terms.intersection(content_terms))

            total = len(query_terms)
            score = overlap / max(total, 1)

            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = score
            scored.append(chunk_copy)

        scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return scored[:top_k]