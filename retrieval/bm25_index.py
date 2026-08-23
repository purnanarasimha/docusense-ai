"""DocuSense AI BM25 Keyword Index
Builds and searches a bm25 indes over all chunks for exat term matching:
    - Invoice numbers, dates, names
    - Technical terms, model names
    - Specific numbers and figures"""

import os
import json
import pickle
import re
from pathlib import Path
from loguru import logger
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

#paths
BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "processed" / "chunks"
INDEX_DIR = BASE_DIR / "data" / "processed" / "bm25_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = INDEX_DIR / "bm25_index.pkl"
CORPUS_FILE = INDEX_DIR / "bm25_corpus.json"

# bm25 index builder
class BM25Index:
    """To build BM25 keyword search index over all doc chunks and fast key word retrieval"""

    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.tokenized = []

    def build(self, chunks_dir: Path = CHUNKS_DIR) -> int:
        """build BM25 index from all chunks files"""
        logger.info("starting bm25 index...")

        chunk_files = list(chunks_dir.glob("*_chunks.json"))

        if not chunk_files:
            logger.error(f"No chunk files in {chunks_dir}")
            return 0

        self.corpus = []
        self.tokenized = []

        for chunk_file in chunk_files:
            with open(chunk_file, encoding="utf-8") as f:
                chunks = json.load(f)

            for chunk in chunks:
                content = chunk.get("content", "")
                if not content.strip():
                    continue

                tokens = self._tokenize(content)

                self.corpus.append(chunk)
                self.tokenized.append(tokens)

        if not self.corpus:
            logger.error("No chunks to index!")
            return 0

        # build index
        self.bm25 = BM25Okapi(self.tokenized)

        self._save()

        logger.success(f"BM25 index build: {len(self.corpus)} chunks")
        return len(self.corpus)

    def _tokenize(self, text: str) -> list:
        """tokenize text for bm 25
        lowercase, remove punctuation, split on whitespace, remove stopwords"""

        # Lowercase
        text = text.lower()

        # Remove special characters but keep numbers
        text = re.sub(r'[^\w\s\d]', ' ', text)

        tokens = text.split()

        stopwords = {'the','a','an','and','or','but','in','on','at',
                     'to','for','of','with','by','from','is','are','was',
                     'were','be','been','being','have','has','had','do',
                     'does','did','will','would','could','should','may','might',
                     'this','that','these','those','it','its','as','if','then',
                     'than','so','etc'}

        tokens = [t for t in tokens if t not in stopwords and len(t)>1]
        return tokens

    # search
    def search(self, query:str, top_k: int=10, doc_id_filter: str = None) -> list:
        """search bm25 index for query returns list of tubles"""

        if not self.bm25:
            loaded = self.load()
            if not loaded:
                logger.error("BM25 index not built yet!")
                return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.info("no query tokens")
            return []

        scores = self.bm25.get_scores(query_tokens)

        scored = [(score, idx) for idx, score in enumerate(scores) if score > 0]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scored[:top_k * 2]:
            chunk = self.corpus[idx]

            if doc_id_filter:
                if chunk.get("doc_id") != doc_id_filter:
                    continue

            results.append({
                "chunk": chunk,
                "bm25_score": float(score),
                "retriever": "bm25",
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "content": chunk.get("content"),
                "content_type": chunk.get("content_type"),
                "page_number": chunk.get("page_number"),
            })

            if len(results) >= top_k:
                break

        return results

    #save or load

    def _save(self):
        """to save bm25 index and corpus to disk"""
        with open(INDEX_FILE, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "tokenized": self.tokenized
            }, f)

        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, indent=2)

        logger.info(f"BM25 index saved : {INDEX_FILE.name}, {CORPUS_FILE.name}")

    def load(self) -> bool:
        """Load BM25 index from disk"""
        if not INDEX_FILE.exists() or not CORPUS_FILE.exists():
            logger.warning("BM25 index files not there. have to run build() first.")
            return False

        try:
            with open(INDEX_FILE, "rb") as f:
                data    = pickle.load(f)
                self.bm25 = data["bm25"]
                self.tokenized = data["tokenized"]

            with open(CORPUS_FILE, encoding="utf-8") as f:
                self.corpus = json.load(f)

            logger.info(f"bm25 index loaded: {len(self.corpus)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def get_stats(self) -> dict:
        """Return index statistics"""
        return{
            "total_chunks": len(self.corpus),
            "avg_doc_length": (
                sum(len(t) for t in self.tokenized)/ max(len(self.tokenized), 1)
            ),
            "unique_docs": len(set(c.get("doc_id") for c in self.corpus)),
        }

if __name__ == "__main__":

    # build index
    index = BM25Index()
    count = index.build()

    print(f"bm25 index bild: {count} chunks")
    print(f"stats: {index.get_stats()}")

    test_queries = [
        "attention mechanism transformer",
        "revenue profit financial",
        "invoice payment total amount",
        "BERT pre-training language model"
    ]

    print("\n"+"="*55)
    print("bm25 search tests")
    print("="*55)

    for query in test_queries:
        results = index.search(query, top_k=2)
        print(f"\n query: '{query}'")
        for i, r in enumerate(results, 1):
            print(f"    {i}. [{r['doc_id']}] score={r['bm25_score']:.3f} | {r['content'][:80]}...")