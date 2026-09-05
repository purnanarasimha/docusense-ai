"""
DocuSense AI - Evaluation Runner
Run full evaluation suite:
  python evals/run_evals.py
  python evals/run_evals.py --quick   (10 questions)
  python evals/run_evals.py --cat financial
"""

import sys
import argparse
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def run_evaluation(
    max_questions : int  = None,
    category      : str  = None,
    sleep_between : float = 1.5
):
    """Run evaluation pipeline"""

    # Import here to avoid slow startup
    from generation.docusense_engine import DocuSenseEngine
    from evals.golden_dataset import (
        GOLDEN_QA_PAIRS,
        print_dataset_summary
    )
    from evals.metrics import DocuSenseEvaluator
    from evals.report_generator import ReportGenerator

    # Print dataset info
    print_dataset_summary()

    # Filter dataset
    qa_pairs = GOLDEN_QA_PAIRS

    if category:
        qa_pairs = [
            q for q in qa_pairs
            if q["category"] == category
        ]
        logger.info(
            f"Filtered to category '{category}': "
            f"{len(qa_pairs)} questions"
        )

    if max_questions:
        qa_pairs = qa_pairs[:max_questions]
        logger.info(
            f"Limited to {max_questions} questions"
        )

    logger.info(
        f"Starting evaluation: "
        f"{len(qa_pairs)} questions"
    )

    # Initialize components
    engine    = DocuSenseEngine(use_gemini_reranker=False)
    evaluator = DocuSenseEvaluator()
    reporter  = ReportGenerator()

    # Run evaluation
    report  = evaluator.evaluate_all(
        engine        = engine,
        qa_pairs      = qa_pairs,
        max_questions = max_questions,
        sleep_between = sleep_between
    )

    # Collect individual results for report
    results = []
    for qa_pair in qa_pairs:
        result = evaluator.evaluate_single(engine, qa_pair)
        results.append(result)

    # Generate reports
    reporter.generate_all(report, results)

    # Return exit code for CI
    if not report.passed_baseline:
        logger.error(
            "Evaluation FAILED baseline thresholds!"
        )
        return 1

    logger.success("Evaluation PASSED all baselines!")
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="DocuSense AI Evaluation Runner"
    )
    parser.add_argument(
        "--quick",
        action  = "store_true",
        help    = "Run quick eval (10 questions)"
    )
    parser.add_argument(
        "--cat",
        type    = str,
        default = None,
        help    = "Filter by category"
    )
    parser.add_argument(
        "--max",
        type    = int,
        default = None,
        help    = "Max questions to evaluate"
    )

    args = parser.parse_args()

    max_q = 10 if args.quick else args.max

    exit_code = run_evaluation(
        max_questions = max_q,
        category      = args.cat,
        sleep_between = 1.5
    )

    sys.exit(exit_code)