"""Answer generator uses gemini to synthesize answers from assembles context"""

import os
import time
from dataclasses import dataclass, asdict
from loguru import logger
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@dataclass
class GeneratedAnswer:
    """Complete answer with metadata"""
    question: str
    answer: str
    confidence: float
    citations: dict
    query_type: str
    doc_ids_used: list
    has_tables: bool
    chunks_used: int
    answer_length: int
    processing_time: float

# promts

SYSTEM_PROMPT = """You are docusense ai, an expert document analysis assistant.
you answer the question that are based only on the Provided document context.

Core Rules:
1. ONLY use information from the provided context
2. If context doesn't contain the answer, say:
    'I cannot find this information in the provided documents.'
3. Always cite sources using [Source N] notation
4. For all the numerical data, quote exact figures from context
5. Be concise but complete
6. If multiple sources disagree, mention both

Response fromat:
-Start with direct answer
-support with evidence from context
-end with confidence note if uncertain"""

QUERY_PROMPTS = {
    "table": """Focus on numerical data and structured information Extract exact figures, percentages, and comparisons from tables, Present data in a clear, organized format.""",
    "factual": """Provide precise, factual answers. Quote exact terms, names, dates from the context. Do not paraphrase when exact wording matters.""",
    "semantic": """Provide a comprehensive explanation. Synthesize information across multiple context chunks. Use clear language to explain concepts.""",
    "hybrid": """Balance precision with comprehension. Include both specific facts and broader context. Organize your answer logically."""
}

class AnswerGenerator:
    """generates answers using gemini with retrieved context"""
    def __init__(self):
        self.model = genai.GenerativeModel(model_name="gemini-3.5-flash-lite", generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=1024, top_p=0.8))
        self.max_retries = 3

    def generate(self, query: str, assembled_ctx: object, citation_bundle: object, query_type: str = "hybrid")->GeneratedAnswer:
        """generated answer from assembled context"""
        start_time = time.time()
        logger.info(f"Generating answer for: '{query[:60]}...")

        # handle empty context
        if not assembled_ctx.context_text:
            return self._no_context_answer(query,start_time)

        # build prompt
        prompt = self._build_prompt(query = query, context = assembled_ctx.context_text, query_type= query_type, has_tables = assembled_ctx.has_tables)

        # generate with gemini
        answer_text = self._call_gemini(prompt)

        # calculate confidenct
        confidence = self._calculate_confidence(answer = answer_text, context = assembled_ctx, citations = citation_bundle)

        processing_time = time.time() - start_time

        result = GeneratedAnswer(
            question= query,
            answer= answer_text,
            confidence= confidence,
            citations= asdict(citation_bundle),
            query_type= query_type,
            doc_ids_used= assembled_ctx.doc_ids_used,
            has_tables= assembled_ctx.has_tables,
            chunks_used= assembled_ctx.chunk_count,
            answer_length= len(answer_text),
            processing_time= round(processing_time, 2)
        )

        logger.success(f"answer generated: {len(answer_text)} chars, confidence={confidence:.2f}, time={processing_time:.2f}s")

        return result

    def _build_prompt(self, query: str, context: str, query_type: str, has_tables: bool) -> str:
        """build optimized prompt for query type"""

        # get query-specific instructions
        query_instructions = QUERY_PROMPTS.get(query_type, QUERY_PROMPTS["hybrid"])

        # table specific addition
        table_note = ""
        if has_tables:
            table_note = ("\nNote: context includes TABLE DATA Prioritize exact figures from tables.")

        prompt = f"""{SYSTEM_PROMPT}
                    {query_instructions}{table_note}
                    
                    =======document context========
                    {context}
                    ==========end context ========
                    Question: {query}
                    answer (cite sources using [Source N])"""
        return prompt

    def _call_gemini(self, prompt: str) -> str:
        """call gemini with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()

            except Exception as e:
                if "429" in str(e):
                    wait = (attempt +1) *10
                    logger.warning(f"rate limit. waiting {wait}s....")
                    time.sleep(wait)
                else:
                    logger.error(f"gemini error: {e}")
                    if attempt == self.max_retries - 1:
                        return ("We encountered error in generating answer, please try again.")
                    time.sleep(5)

        return "Failed to generate answer after retries"

    def _calculate_confidence(self, answer: str, context: object, citations: object)-> float:
        """calclate confidence score between 0 and 1"""

        score = 0.5 # base score

        # more sources = higher cofidence
        source_count = citations.total_sources
        if source_count >=3:
            score += 0.2
        elif source_count >=1:
            score += 0.1

        # table sources = higher confidexce for data questions
        if citations.has_table_source:
            score += 0.1

        # penalize uncertain answers
        uncertain_phrases = ["cannot find", "not mentioned", "not available", "unclear","i don't know", "no information"]
        if any(p in answer.lower() for p in uncertain_phrases):
            score -=0.3

        # reward log answers
        if len(answer)> 300:
            score +=0.1

        # multiple docs used more comprehensive
        if len(context.doc_ids_used) > 1:
            score += 0.1

        # clamp between 0 and 1
        return round(min(max(score, 0.0), 1.0), 2)

    def _no_context_answer(self, query: str, start_time: float) -> GeneratedAnswer:
        """return when no context available"""
        from generation.ciration_builder import(CitationBundle)

        empty_bundle = CitationBundle(
            citations= [],
            total_sources= 0,
            unique_docs= [],
            has_table_source=False,
            has_image_source= False
        )

        return GeneratedAnswer(
            question= query,
            answer= "I can not find relevant information in the provided documents to answer your question",
            confidence= 0.0,
            citations= asdict(empty_bundle),
            query_type= "unknown",
            doc_ids_used= [],
            has_tables= False,
            chunks_used= 0,
            answer_length= 0,
            processing_time= round(time.time() - start_time, 2)
        )