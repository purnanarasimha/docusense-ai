"""Master engine to orchestrates the full RAG PIPELINE
This is main entry point fro all question answering"""

import time
from dataclasses import dataclass,asdict
from loguru import logger
from dotenv import load_dotenv

from retrieval.hybrid_retriever import HybridRetriever
from generation.context_assembler import ContextAssembler
from generation.ciration_builder import CitationBuilder
from generation.answer_generator import AnswerGenerator

load_dotenv()

@dataclass
class DocuSenseResponse:
    """conplete rag response"""
    question: str
    answer: str
    confidence: float
    citations: dict
    retrieval_stats: dict
    doc_ids_used: list
    query_type: str
    has_tables: bool
    chunks_used: int
    processing_time: float
    status: str

# Docusense engine

class DocuSenseEngine:
    """Master Rag orchestrator single class to rule them all"""

    def __init__(self, use_gemini_reranker: bool = False):
        logger.info("Initializing DocuSense engine...")

        self.retriever = HybridRetriever(use_gemini_reranker=use_gemini_reranker)
        self.assembler = ContextAssembler()
        self.citer = CitationBuilder()
        self.generator = AnswerGenerator()

        logger.success("DocuSense Engine ready!")

    def query(self, question: str, top_k: int = 5, doc_id_filter: str = None) -> DocuSenseResponse:
        """Main query method - full rag pipeline"""
        start_time = time.time()
        logger.info(f"\nProcesssing query: '{question}'")

        try:
            retrieval_result = self.retriever.retrieve(query=question, top_k=top_k, doc_id_filter= doc_id_filter)

            chunks = retrieval_result["chunks"]
            query_type = retrieval_result["query_type"]
            ret_stats = retrieval_result["retrieval_stats"]

            logger.info(f"Retrieved {len(chunks)} chunks (type={query_type})")

            context = self.assembler.assemble(chunks=chunks, query=question, doc_filter=doc_id_filter)

            citations = self.citer.build(source_chunks=context.source_chunks)

            generated = self.generator.generate(query= question, assembled_ctx= context, citation_bundle= citations, query_type= query_type)

            processing_time = time.time() - start_time

            response = DocuSenseResponse (
                question= question,
                answer= generated.answer,
                confidence= generated.confidence,
                citations= generated.citations,
                retrieval_stats= ret_stats,
                doc_ids_used= context.doc_ids_used,
                query_type= query_type,
                has_tables= context.has_tables,
                chunks_used= context.chunk_count,
                processing_time= round(processing_time, 2),
                status= "success"
            )

            logger.success(f"Query complete: confidence={generated.confidence}, time={processing_time:.2f}s")

            return response

        except Exception as e:
            logger.error(f"Engine error: {e}")
            return self._error_response(question, str(e), time.time()-start_time)

    def _error_response(self, question: str, error: str, elapsed: float) -> DocuSenseResponse:
        return DocuSenseResponse(
            question= question,
            answer= f"Error processing query: {error}",
            confidence= 0.0,
            citations= {},
            retrieval_stats= {},
            doc_ids_used= [],
            query_type= "unkown",
            has_tables= False,
            chunks_used= 0,
            processing_time= round(elapsed,2),
            status= "error"
        )

    def display_response(self, response: DocuSenseResponse) -> None:
        """Pretty print response to terminal"""
        print("\n" + "=" * 65)
        print(" Docusense AI response")
        print("="*65)
        print(f"\n Q: {response.question}")
        print("\n" + "-"*65)
        print(f"\n Answer:\n")

        # word wrap answer
        words = response.answer.split()
        line = " "
        for word in words:
            if len(line) + len(word) > 70:
                print(line)
                line = " " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

        print("\n" + "-" *65)
        print(f"\n confidence: {response.confidence:.0%}")
        print(f" Query type : {response.query_type}")
        print(f" docs used : {response.doc_ids_used}")
        print(f" chunks used : {response.chunks_used}")
        print(f" has tables : {response.has_tables}")
        print(f" Process time : {response.processing_time}")
        print(f" Status : {response.status}")

        # show citations
        citations = response.citations.get("citations", [])
        if citations:
            print(f"\n Sources ({len(citations)})")
            for cite in citations:
                doc = cite['doc_id'].replace('_',' ').title()
                ctype = cite['content_type']
                page = cite['page_number']
                print(f"[{cite['citation_number']}] {doc}-page {page} ({ctype})")

        # show retrieval stats
        stats = response.retrieval_stats
        if stats:
            print(f"\n Retrieval Stats:")
            print(f"    Semantic: {stats.get('semantic_found', 0)}")
            print(f"    BM25: {stats.get('bm25_found', 0)}")
            print(f"    fused: {stats.get('after fusion', 0)}")
            print(f"    Final: {stats.get('after_rerank', 0)}")

        print("\n" + "=" * 65)

# run tests
if __name__ == "__main__":
    engine = DocuSenseEngine(use_gemini_reranker=False)

    # test queries covering all doc types
    test_queries = [
        {
            "question": "what is the attention mechanism and how does it work",
            "expected_doc": "attention_is_all_you_need"
        },
        {
            "question": "what was total revenue and net income for the year for techventure?",
            "expected_doc": "financial_report_techventure_2023"
        },
        {
            "question": "what are the payment terms and total amount in the invoice?",
            "expected_doc": "invoice_inv_2024_001"
        },
        {
            "question": "what is bert and how it is from gpt?",
            "expected_doc": "bert_paper"
        }
    ]

    print("\n" + "="*20)
    print(" DOCUSENSE AI FULL RAG PIPE LINE TEST")
    print("="*60)

    results = []

    for i, test in enumerate(test_queries, 1):
        print(f"\n[Test {i}/{len(test_queries)}]")
        response = engine.query(question= test["question"], top_k=5)
        engine.display_response(response)

        results.append({
            "question": test["question"],
            "expected_doc": test["expected_doc"],
            "got_docs": response.doc_ids_used,
            "confidence": response.confidence,
            "status": response.status,
            "correct_doc": test["expected_doc"] in (response.doc_ids_used or [])
        })

    # final summary
    print("\n" + "=" * 65)
    print(" TEST SUMMARY")
    print("=" * 65)

    correct = sum(1 for r in results if r["correct_doc"])

    for r in results:
        icon = "OK" if r["correct_doc"] else "NO"
        print(f"\n {icon} Q: {r['question'][:45]}...")
        print(f"    Expected: {r['expected_doc']}")
        print(f"    Got: {r['got_docs']}")
        print(f"    confidence: {r['confidence']:.0%}")

    print("\n" + "="*65)
    print(f"correct doc retrieved: {correct}/{len(results)}")
    print("="*65)