"""
DocuSense AI - Custom Evaluation Metrics
Beyond RAGAS - measures what matters
for production RAG systems
"""

import time
import re
from dataclasses import dataclass
from loguru import logger


# ===================================
# Data Models
# ===================================

@dataclass
class EvalResult:
    """Result for a single question"""
    question            : str
    answer              : str
    ground_truth        : str
    category            : str
    difficulty          : str
    doc_id              : str

    # Retrieval metrics
    correct_doc_found   : bool
    chunks_used         : int
    has_citations       : bool
    citation_count      : int

    # Quality metrics
    confidence          : float
    processing_time     : float
    answer_length       : int

    # Custom scores
    keyword_overlap     : float
    length_adequacy     : float

    # Status
    status              : str
    error               : str = ""


@dataclass
class EvalReport:
    """Aggregated evaluation report"""
    total_questions     : int
    successful          : int
    failed              : int

    # Retrieval
    correct_doc_rate    : float
    avg_chunks_used     : float
    citation_rate       : float

    # Quality
    avg_confidence      : float
    avg_processing_time : float
    avg_keyword_overlap : float
    avg_length_adequacy : float

    # By category
    category_scores     : dict
    difficulty_scores   : dict

    # Baseline comparison
    passed_baseline     : bool
    baseline_metrics    : dict


# ===================================
# Metric Calculators
# ===================================

class MetricCalculator:
    """
    Calculates custom evaluation metrics
    No external API needed for these
    """

    def keyword_overlap(
        self,
        answer      : str,
        ground_truth: str
    ) -> float:
        """
        Measure keyword overlap between
        answer and ground truth
        Range: 0.0 - 1.0
        """
        if not answer or not ground_truth:
            return 0.0

        # Tokenize
        ans_tokens = set(
            re.sub(r'[^\w\s]', '', answer.lower()).split()
        )
        gt_tokens  = set(
            re.sub(r'[^\w\s]', '', ground_truth.lower()).split()
        )

        # Remove stopwords
        stops = {
            'the', 'a', 'an', 'and', 'or', 'is',
            'are', 'was', 'were', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'it', 'this'
        }
        ans_tokens -= stops
        gt_tokens  -= stops

        if not gt_tokens:
            return 0.0

        overlap = ans_tokens.intersection(gt_tokens)
        return round(len(overlap) / len(gt_tokens), 3)

    def length_adequacy(
        self,
        answer      : str,
        ground_truth: str
    ) -> float:
        """
        Check if answer length is appropriate
        relative to ground truth
        Range: 0.0 - 1.0
        """
        if not answer:
            return 0.0

        ans_words = len(answer.split())
        gt_words  = len(ground_truth.split())

        if gt_words == 0:
            return 0.0

        ratio = ans_words / gt_words

        # Ideal ratio: 0.5 to 3.0
        if 0.5 <= ratio <= 3.0:
            return 1.0
        elif ratio < 0.5:
            return ratio / 0.5
        else:
            return max(0.0, 1.0 - (ratio - 3.0) / 10)

    def citation_quality(
        self,
        citations   : dict
    ) -> dict:
        """
        Evaluate citation quality
        Returns quality metrics dict
        """
        citation_list = citations.get("citations", [])

        if not citation_list:
            return {
                "has_citations"  : False,
                "citation_count" : 0,
                "has_table_src"  : False,
                "unique_docs"    : 0
            }

        return {
            "has_citations"  : True,
            "citation_count" : len(citation_list),
            "has_table_src"  : citations.get(
                "has_table_source", False
            ),
            "unique_docs"    : len(
                citations.get("unique_docs", [])
            )
        }

    def correctness_check(
        self,
        answer      : str,
        ground_truth: str
    ) -> float:
        """
        Check if key facts from ground truth
        appear in the answer
        Range: 0.0 - 1.0
        """
        if not answer or not ground_truth:
            return 0.0

        # Extract numbers and key terms from ground truth
        numbers = re.findall(
            r'\b\d+(?:[.,]\d+)?(?:\s?(?:billion|million|'
            r'thousand|percent|%))?\b',
            ground_truth.lower()
        )

        if not numbers:
            return self.keyword_overlap(answer, ground_truth)

        # Check how many numbers appear in answer
        found = sum(
            1 for num in numbers
            if num.lower() in answer.lower()
        )

        return round(found / len(numbers), 3) if numbers else 0.0


# ===================================
# Evaluator
# ===================================

class DocuSenseEvaluator:
    """
    Runs full evaluation suite
    against the golden dataset
    """

    # Baseline thresholds for CI
    BASELINES = {
        "correct_doc_rate"    : 0.60,
        "citation_rate"       : 0.70,
        "avg_confidence"      : 0.40,
        "avg_keyword_overlap" : 0.25,
        "avg_length_adequacy" : 0.60,
    }

    def __init__(self):
        self.calculator = MetricCalculator()

    def evaluate_single(
        self,
        engine,
        qa_pair : dict
    ) -> EvalResult:
        """Evaluate one Q&A pair"""

        question     = qa_pair["question"]
        ground_truth = qa_pair["ground_truth"]
        expected_doc = qa_pair.get("doc_id")
        category     = qa_pair.get("category", "general")
        difficulty   = qa_pair.get("difficulty", "medium")

        try:
            # Query the engine
            response = engine.query(
                question = question,
                top_k    = 5
            )

            answer    = response.answer
            citations = response.citations

            # Check correct doc retrieved
            docs_used       = response.doc_ids_used or []
            correct_doc     = (
                expected_doc in docs_used
                if expected_doc else True
            )

            # Citation metrics
            cite_quality    = self.calculator.citation_quality(
                citations
            )

            # Quality metrics
            kw_overlap      = self.calculator.keyword_overlap(
                answer, ground_truth
            )
            len_adequacy    = self.calculator.length_adequacy(
                answer, ground_truth
            )

            return EvalResult(
                question          = question,
                answer            = answer,
                ground_truth      = ground_truth,
                category          = category,
                difficulty        = difficulty,
                doc_id            = expected_doc or "any",
                correct_doc_found = correct_doc,
                chunks_used       = response.chunks_used,
                has_citations     = cite_quality["has_citations"],
                citation_count    = cite_quality["citation_count"],
                confidence        = response.confidence,
                processing_time   = response.processing_time,
                answer_length     = len(answer.split()),
                keyword_overlap   = kw_overlap,
                length_adequacy   = len_adequacy,
                status            = "success"
            )

        except Exception as e:
            logger.error(f"Eval failed for '{question}': {e}")
            return EvalResult(
                question          = question,
                answer            = "",
                ground_truth      = ground_truth,
                category          = category,
                difficulty        = difficulty,
                doc_id            = expected_doc or "any",
                correct_doc_found = False,
                chunks_used       = 0,
                has_citations     = False,
                citation_count    = 0,
                confidence        = 0.0,
                processing_time   = 0.0,
                answer_length     = 0,
                keyword_overlap   = 0.0,
                length_adequacy   = 0.0,
                status            = "failed",
                error             = str(e)
            )

    def evaluate_all(
        self,
        engine,
        qa_pairs    : list,
        max_questions: int = None,
        sleep_between: float = 1.0
    ) -> EvalReport:
        """
        Run evaluation on all Q&A pairs
        Returns aggregated EvalReport
        """
        if max_questions:
            qa_pairs = qa_pairs[:max_questions]

        logger.info(
            f"Running evaluation on "
            f"{len(qa_pairs)} questions..."
        )

        results = []

        for i, qa_pair in enumerate(qa_pairs, 1):
            logger.info(
                f"[{i}/{len(qa_pairs)}] "
                f"{qa_pair['question'][:50]}..."
            )

            result = self.evaluate_single(engine, qa_pair)
            results.append(result)

            # Progress update
            if result.status == "success":
                logger.info(
                    f"  ✅ overlap={result.keyword_overlap:.2f} "
                    f"conf={result.confidence:.2f} "
                    f"correct_doc={result.correct_doc_found}"
                )
            else:
                logger.warning(f"  ❌ {result.error}")

            time.sleep(sleep_between)

        return self._aggregate(results)

    def _aggregate(self, results: list) -> EvalReport:
        """Aggregate individual results into report"""
        total      = len(results)
        successful = [r for r in results if r.status == "success"]
        failed     = [r for r in results if r.status == "failed"]

        if not successful:
            logger.error("No successful evaluations!")
            return self._empty_report(total)

        # Core metrics
        correct_doc_rate = sum(
            1 for r in successful if r.correct_doc_found
        ) / len(successful)

        citation_rate = sum(
            1 for r in successful if r.has_citations
        ) / len(successful)

        avg_confidence = sum(
            r.confidence for r in successful
        ) / len(successful)

        avg_time = sum(
            r.processing_time for r in successful
        ) / len(successful)

        avg_kw_overlap = sum(
            r.keyword_overlap for r in successful
        ) / len(successful)

        avg_len_adequacy = sum(
            r.length_adequacy for r in successful
        ) / len(successful)

        avg_chunks = sum(
            r.chunks_used for r in successful
        ) / len(successful)

        # By category
        categories = set(r.category for r in successful)
        cat_scores = {}
        for cat in categories:
            cat_results = [
                r for r in successful if r.category == cat
            ]
            cat_scores[cat] = {
                "count"         : len(cat_results),
                "keyword_overlap": round(sum(
                    r.keyword_overlap for r in cat_results
                ) / len(cat_results), 3),
                "confidence"    : round(sum(
                    r.confidence for r in cat_results
                ) / len(cat_results), 3),
                "correct_doc"   : round(sum(
                    1 for r in cat_results
                    if r.correct_doc_found
                ) / len(cat_results), 3),
            }

        # By difficulty
        difficulties = set(r.difficulty for r in successful)
        diff_scores  = {}
        for diff in difficulties:
            diff_results = [
                r for r in successful if r.difficulty == diff
            ]
            diff_scores[diff] = {
                "count"         : len(diff_results),
                "keyword_overlap": round(sum(
                    r.keyword_overlap for r in diff_results
                ) / len(diff_results), 3),
                "confidence"    : round(sum(
                    r.confidence for r in diff_results
                ) / len(diff_results), 3),
            }

        # Check baselines
        current = {
            "correct_doc_rate"    : correct_doc_rate,
            "citation_rate"       : citation_rate,
            "avg_confidence"      : avg_confidence,
            "avg_keyword_overlap" : avg_kw_overlap,
            "avg_length_adequacy" : avg_len_adequacy,
        }

        passed_baseline = all(
            current[metric] >= threshold
            for metric, threshold in self.BASELINES.items()
        )

        return EvalReport(
            total_questions     = total,
            successful          = len(successful),
            failed              = len(failed),
            correct_doc_rate    = round(correct_doc_rate, 3),
            avg_chunks_used     = round(avg_chunks, 1),
            citation_rate       = round(citation_rate, 3),
            avg_confidence      = round(avg_confidence, 3),
            avg_processing_time = round(avg_time, 2),
            avg_keyword_overlap = round(avg_kw_overlap, 3),
            avg_length_adequacy = round(avg_len_adequacy, 3),
            category_scores     = cat_scores,
            difficulty_scores   = diff_scores,
            passed_baseline     = passed_baseline,
            baseline_metrics    = self.BASELINES
        )

    def _empty_report(self, total: int) -> EvalReport:
        return EvalReport(
            total_questions     = total,
            successful          = 0,
            failed              = total,
            correct_doc_rate    = 0.0,
            avg_chunks_used     = 0.0,
            citation_rate       = 0.0,
            avg_confidence      = 0.0,
            avg_processing_time = 0.0,
            avg_keyword_overlap = 0.0,
            avg_length_adequacy = 0.0,
            category_scores     = {},
            difficulty_scores   = {},
            passed_baseline     = False,
            baseline_metrics    = self.BASELINES
        )