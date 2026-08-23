"""reciprocal rank fusion combine results from multiple retrievers into a single ranked list"""

from loguru import logger

# rrf fusion

class RRFFusion:
    """reciprocal rank fusion algorithm merges semantic + bm25 results into unified ranked list
    it is because - no need to noramlize scores across retrievers robust to outliers, provert o out perform individual retrievers"""

    def __init__(self, k: int = 60):
        """k 60 is standard rrf constant higher k reduces impact of top ranks"""
        self.k = k

    def fuse(self, result_lists: list, weights: list = None) -> list:
        """fuse multiple result lists using RRF
        args: result_lists: list of result list each result must have chunk id
              weights: Optional weights per retriever Default equal weights
        returns: single sorter list with rrf_score"""

        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        # normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # calculate rrf score
        rrf_scores = {}
        chunk_data = {}

        for list_idx, result_list in enumerate(result_lists):
            weight = weights[list_idx]

            for rank, result in enumerate(result_list):
                chunk_id = result.get("chunk_id")

                if not chunk_id:
                    continue

                # rrf formula : weight / (k + rank)
                rrf_score = weight / (self.k + rank)

                # accumulate scores
                if chunk_id in rrf_scores:
                    rrf_scores[chunk_id] += rrf_score
                else:
                    rrf_scores[chunk_id] = rrf_score
                    chunk_data[chunk_id] = result

        # build fused results
        fused = []
        for chunk_id, rrf_score in rrf_scores.items():
            result = chunk_data[chunk_id].copy()
            result["rrf_score"] = rrf_score

            # track which retrievers found this chunks
            retrievers = []
            for result_list in result_lists:
                for r in result_list:
                    if r.get("chunk_id") == chunk_id:
                        retrievers.append(r.get("retrievert", "unknown"))
            result["found_by"] = list(set(retrievers))
            fused.append(result)

        # sort by rrf score descending
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        return fused

    def fuse_semantic_and_bm25(self, semantic_results: list, bm25_results: list, semantic_weight: float = 0.7, bm25_weight: float = 0.3) -> list:
        """convenience method to fuse semantic + bm25 results Semantic gets higher weight by default"""
        return self.fuse(result_lists=[semantic_results, bm25_results], weights=[semantic_weight, bm25_weight])


# query router

class QueryRouter:
    """routes queries to appropriate retrieval stategy based on query type detection"""

    def classify_query(self, query: str) -> str:
        """Classify query type to determise best retrieval strategy 
        Returns:    'factual' - bm25 weighted higher
                    'semantic' - semantic weighted higher
                    'table' - search tables first
                    'hibrid' - equal weights by default"""

        query_lower = query.lower()

        # table/numeric queries
        table_keywords = ['how much', 'total', 'revenue', 'profit', 'cost', 'price', 'amount', 'number of',
                          'how many', 'percentage', 'rate', 'ratio', 'compare', 'versus', 'vs', 'difference',
                          'increase', 'decrease', 'growth']

        # factual queries
        factual_keywords = [
            'what is', 'who is', 'when', 'where',
            'which', 'invoice', 'contract', 'date',
            'name', 'number', 'id', 'ref'
        ]

        # conceptual queries
        semantic_keywords = [
            'explain', 'how does', 'why', 'describe',
            'summarize', 'overview', 'concept', 'approact',
            'method', 'technique'
        ]

        # score each type
        table_score = sum(1 for kw in table_keywords if kw in query_lower)
        factual_score = sum(1 for kw in factual_keywords if kw in query_lower)
        semantic_score = sum(1 for kw in semantic_keywords if kw in query_lower)

        # return dominant type
        if table_score >=2:
            return "table"
        elif factual_score > semantic_score:
            return "factual"
        elif semantic_score > factual_score:
            return "semantic"
        else:
            return "hybrid"

    def get_weights(self, query_type: str) -> dict:
        """get retriever weights for query type"""
        weights = {
            "table": {
                "semantic": 0.4,
                "bm25": 0.6
            },
            "factual": {
                "semantic": 0.5,
                "bm25": 0.6
            },
            "semantic": {
                "semantic": 0.8,
                "bm25": 0.2
            },
            "hybrid": {
                "semantic": 0.7,
                "bm25": 0.3
            }
        }
        return weights.get(query_type, weights['hybrid'])