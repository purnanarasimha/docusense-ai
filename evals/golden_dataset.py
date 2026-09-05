"""
DocuSense AI - Golden Test Dataset
50 manually crafted Q&A pairs
covering all document types
Used for RAGAS evaluation
"""

from datasets import Dataset

# ===================================
# Golden Q&A Pairs
# ===================================

GOLDEN_QA_PAIRS = [

    # ── Research Papers ─────────────────

    {
        "question": (
            "What is the attention mechanism "
            "in transformer models?"
        ),
        "ground_truth": (
            "The attention mechanism allows transformer "
            "models to weigh the importance of different "
            "parts of the input sequence when generating "
            "output. It computes a weighted sum of values "
            "based on the similarity between queries and "
            "keys using scaled dot-product attention."
        ),
        "doc_id"    : "attention_is_all_you_need",
        "category"  : "research_paper",
        "difficulty": "medium"
    },
    {
        "question": (
            "What does the acronym BERT stand for?"
        ),
        "ground_truth": (
            "BERT stands for Bidirectional Encoder "
            "Representations from Transformers."
        ),
        "doc_id"    : "bert_paper",
        "category"  : "research_paper",
        "difficulty": "easy"
    },
    {
        "question": (
            "What is multi-head attention and "
            "why is it used?"
        ),
        "ground_truth": (
            "Multi-head attention runs the attention "
            "mechanism multiple times in parallel with "
            "different learned projections. This allows "
            "the model to jointly attend to information "
            "from different representation subspaces "
            "at different positions."
        ),
        "doc_id"    : "attention_is_all_you_need",
        "category"  : "research_paper",
        "difficulty": "medium"
    },
    {
        "question": (
            "What training objective does BERT use?"
        ),
        "ground_truth": (
            "BERT uses two training objectives: "
            "Masked Language Modeling (MLM) where "
            "random tokens are masked and predicted, "
            "and Next Sentence Prediction (NSP) where "
            "the model predicts if two sentences "
            "are consecutive."
        ),
        "doc_id"    : "bert_paper",
        "category"  : "research_paper",
        "difficulty": "medium"
    },
    {
        "question": (
            "What is positional encoding in transformers?"
        ),
        "ground_truth": (
            "Positional encoding adds information about "
            "the position of tokens in the sequence "
            "since transformers have no inherent notion "
            "of order. It uses sine and cosine functions "
            "of different frequencies to create unique "
            "position representations."
        ),
        "doc_id"    : "attention_is_all_you_need",
        "category"  : "research_paper",
        "difficulty": "medium"
    },
    {
        "question": (
            "What problem does RAG solve in "
            "language model applications?"
        ),
        "ground_truth": (
            "RAG solves the problem of language models "
            "having outdated or missing knowledge by "
            "combining retrieval of relevant documents "
            "with generation. This allows models to "
            "access up-to-date information and provide "
            "more accurate, grounded responses."
        ),
        "doc_id"    : "rag_paper",
        "category"  : "research_paper",
        "difficulty": "medium"
    },
    {
        "question": (
            "What are the two components of "
            "a RAG system?"
        ),
        "ground_truth": (
            "A RAG system has two main components: "
            "a retriever that finds relevant documents "
            "from a knowledge base, and a generator "
            "that produces the final answer by "
            "conditioning on both the query and "
            "retrieved documents."
        ),
        "doc_id"    : "rag_paper",
        "category"  : "research_paper",
        "difficulty": "easy"
    },
    {
        "question": (
            "How many parameters does the "
            "transformer base model have?"
        ),
        "ground_truth": (
            "The transformer base model has "
            "65 million parameters."
        ),
        "doc_id"    : "attention_is_all_you_need",
        "category"  : "research_paper",
        "difficulty": "hard"
    },
    {
        "question": (
            "What tokenization does BERT use?"
        ),
        "ground_truth": (
            "BERT uses WordPiece tokenization which "
            "splits words into subword units. It has "
            "a vocabulary of 30,000 tokens and uses "
            "special tokens like [CLS] at the start "
            "and [SEP] to separate sentences."
        ),
        "doc_id"    : "bert_paper",
        "category"  : "research_paper",
        "difficulty": "medium"
    },
    {
        "question": (
            "What is the key difference between "
            "encoder and decoder in transformers?"
        ),
        "ground_truth": (
            "The encoder processes the input sequence "
            "and creates representations using "
            "bidirectional self-attention. The decoder "
            "generates output tokens sequentially "
            "using masked self-attention to prevent "
            "attending to future positions."
        ),
        "doc_id"    : "attention_is_all_you_need",
        "category"  : "research_paper",
        "difficulty": "medium"
    },

    # ── Financial Reports ────────────────

    {
        "question": (
            "What was the total revenue for "
            "TechVenture in 2023?"
        ),
        "ground_truth": (
            "TechVenture Corporation's total revenue "
            "for fiscal year 2023 was 4.2 billion "
            "dollars, representing a 23 percent "
            "increase year-over-year."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "easy"
    },
    {
        "question": (
            "What was TechVenture's net income "
            "in 2023?"
        ),
        "ground_truth": (
            "TechVenture's net income for 2023 "
            "was 672 million dollars, representing "
            "a 16 percent net margin."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "easy"
    },
    {
        "question": (
            "How much did TechVenture spend "
            "on research and development?"
        ),
        "ground_truth": (
            "TechVenture spent 630 million dollars "
            "on research and development in 2023, "
            "which represents 15 percent of total "
            "revenue. They filed 847 patents and "
            "launched 12 new products."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "medium"
    },
    {
        "question": (
            "What was TechVenture's gross profit "
            "margin in 2023?"
        ),
        "ground_truth": (
            "TechVenture achieved a gross profit "
            "of 2.1 billion dollars in 2023, "
            "representing a 50 percent gross margin."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "easy"
    },
    {
        "question": (
            "What percentage of revenue came "
            "from North America?"
        ),
        "ground_truth": (
            "North America contributed 45 percent "
            "of TechVenture's revenue, amounting "
            "to 1.89 billion dollars in 2023."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "medium"
    },
    {
        "question": (
            "What is TechVenture's market share "
            "and market position?"
        ),
        "ground_truth": (
            "TechVenture holds the number 2 position "
            "in enterprise software globally with a "
            "market share of 18.3 percent, up from "
            "15.1 percent in 2022."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "medium"
    },
    {
        "question": (
            "What revenue guidance did TechVenture "
            "provide for 2024?"
        ),
        "ground_truth": (
            "TechVenture provided 2024 revenue "
            "guidance of 5.0 to 5.2 billion dollars, "
            "representing 19 to 24 percent growth. "
            "EPS guidance was 5.00 to 5.20 dollars "
            "diluted."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "medium"
    },
    {
        "question": (
            "What are TechVenture's total assets?"
        ),
        "ground_truth": (
            "TechVenture's total assets as of "
            "December 31 2023 were 17.05 billion "
            "dollars, including 5.75 billion in "
            "current assets and 11.3 billion in "
            "non-current assets."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "hard"
    },
    {
        "question": (
            "What was Q4 2023 revenue for TechVenture?"
        ),
        "ground_truth": (
            "TechVenture's Q4 2023 revenue was "
            "1.12 billion dollars, representing "
            "28 percent growth year-over-year."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "medium"
    },
    {
        "question": (
            "What is CloudSuite Enterprise revenue "
            "and percentage of total?"
        ),
        "ground_truth": (
            "CloudSuite Enterprise generated "
            "1.8 billion dollars in revenue, "
            "representing 43 percent of "
            "TechVenture's total revenue."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "hard"
    },

    # ── Invoice ──────────────────────────

    {
        "question": (
            "What is the total amount on the invoice?"
        ),
        "ground_truth": (
            "The total amount on invoice INV-2024-001 "
            "is 11,440 dollars, which includes a "
            "subtotal of 10,400 dollars plus "
            "10 percent tax of 1,040 dollars."
        ),
        "doc_id"    : "invoice_inv_2024_001",
        "category"  : "invoice",
        "difficulty": "easy"
    },
    {
        "question": (
            "What are the payment terms on the invoice?"
        ),
        "ground_truth": (
            "The invoice payment terms are Net 30, "
            "meaning payment is due within 30 days. "
            "Payment should be made via bank transfer "
            "and include invoice number INV-2024-001."
        ),
        "doc_id"    : "invoice_inv_2024_001",
        "category"  : "invoice",
        "difficulty": "easy"
    },
    {
        "question": (
            "How many hours of software development "
            "were billed?"
        ),
        "ground_truth": (
            "40 hours of software development were "
            "billed at 150 dollars per hour, "
            "totaling 6,000 dollars."
        ),
        "doc_id"    : "invoice_inv_2024_001",
        "category"  : "invoice",
        "difficulty": "medium"
    },
    {
        "question": (
            "Who is the invoice billed to?"
        ),
        "ground_truth": (
            "The invoice is billed to Acme Corporation "
            "at 123 Business Street, New York, NY 10001."
        ),
        "doc_id"    : "invoice_inv_2024_001",
        "category"  : "invoice",
        "difficulty": "easy"
    },
    {
        "question": (
            "What is the invoice number and date?"
        ),
        "ground_truth": (
            "The invoice number is INV-2024-001 "
            "dated January 15 2024 with a due "
            "date of February 15 2024."
        ),
        "doc_id"    : "invoice_inv_2024_001",
        "category"  : "invoice",
        "difficulty": "easy"
    },
    {
        "question": (
            "What was the UI design cost?"
        ),
        "ground_truth": (
            "UI design was billed at 20 hours "
            "at 120 dollars per hour, "
            "totaling 2,400 dollars."
        ),
        "doc_id"    : "invoice_inv_2024_001",
        "category"  : "invoice",
        "difficulty": "medium"
    },
    {
        "question": (
            "What bank details are provided "
            "for payment?"
        ),
        "ground_truth": (
            "Payment should be made to First National "
            "Bank, account number 1234567890, "
            "routing number 021000021."
        ),
        "doc_id"    : "invoice_inv_2024_001",
        "category"  : "invoice",
        "difficulty": "medium"
    },

    # ── Contract ─────────────────────────

    {
        "question": (
            "What is the total value of "
            "the software contract?"
        ),
        "ground_truth": (
            "The total contract value is "
            "120,000 dollars for software "
            "development services across "
            "three phases."
        ),
        "doc_id"    : "software_contract_2024",
        "category"  : "contract",
        "difficulty": "easy"
    },
    {
        "question": (
            "What are the three phases of "
            "the contract and their costs?"
        ),
        "ground_truth": (
            "Phase 1 Requirements and Design is "
            "25,000 dollars over 30 days. "
            "Phase 2 Development and Testing is "
            "75,000 dollars over 90 days. "
            "Phase 3 Deployment and Training is "
            "20,000 dollars over 30 days."
        ),
        "doc_id"    : "software_contract_2024",
        "category"  : "contract",
        "difficulty": "medium"
    },
    {
        "question": (
            "What is the upfront payment "
            "required to start the contract?"
        ),
        "ground_truth": (
            "30 percent upfront payment is required "
            "upon signing, which equals "
            "36,000 dollars."
        ),
        "doc_id"    : "software_contract_2024",
        "category"  : "contract",
        "difficulty": "easy"
    },
    {
        "question": (
            "Who are the parties in the "
            "software development agreement?"
        ),
        "ground_truth": (
            "The client is Acme Corporation, "
            "a Delaware corporation. The developer "
            "is TechCorp Solutions LLC, "
            "a New York LLC."
        ),
        "doc_id"    : "software_contract_2024",
        "category"  : "contract",
        "difficulty": "easy"
    },
    {
        "question": (
            "What is the contract term and "
            "termination notice period?"
        ),
        "ground_truth": (
            "The contract runs from January 1 2024 "
            "to June 30 2024. Either party can "
            "terminate with 30 days written notice."
        ),
        "doc_id"    : "software_contract_2024",
        "category"  : "contract",
        "difficulty": "medium"
    },
    {
        "question": (
            "Who owns the intellectual property "
            "created under the contract?"
        ),
        "ground_truth": (
            "All work product, code and deliverables "
            "created under the agreement shall be "
            "owned exclusively by the client "
            "upon full payment."
        ),
        "doc_id"    : "software_contract_2024",
        "category"  : "contract",
        "difficulty": "medium"
    },
    {
        "question": (
            "What is the late payment penalty?"
        ),
        "ground_truth": (
            "The late payment penalty is "
            "1.5 percent per month on "
            "outstanding balances."
        ),
        "doc_id"    : "software_contract_2024",
        "category"  : "contract",
        "difficulty": "medium"
    },

    # ── Cross-document ───────────────────

    {
        "question": (
            "What AI or machine learning technologies "
            "are mentioned across all documents?"
        ),
        "ground_truth": (
            "The documents mention transformer "
            "architecture, attention mechanisms, "
            "BERT, RAG systems, and AI integration "
            "in enterprise software products."
        ),
        "doc_id"    : None,
        "category"  : "cross_document",
        "difficulty": "hard"
    },
    {
        "question": (
            "Which documents contain financial "
            "or payment information?"
        ),
        "ground_truth": (
            "The financial report contains revenue "
            "and profit data. The invoice contains "
            "payment terms and amounts. The contract "
            "contains payment schedules and totals."
        ),
        "doc_id"    : None,
        "category"  : "cross_document",
        "difficulty": "medium"
    },
    {
        "question": (
            "What is the total deployment cost "
            "in the contract?"
        ),
        "ground_truth": (
            "The deployment and training phase "
            "costs 20,000 dollars over 30 days, "
            "billed at 200 dollars per hour "
            "for 5 hours plus training costs."
        ),
        "doc_id"    : "software_contract_2024",
        "category"  : "contract",
        "difficulty": "hard"
    },
    {
        "question": (
            "What is the operating income margin "
            "for TechVenture?"
        ),
        "ground_truth": (
            "TechVenture's operating income was "
            "840 million dollars representing "
            "a 20 percent operating margin "
            "on total revenue of 4.2 billion."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "hard"
    },
    {
        "question": (
            "What testing hours are billed "
            "in the invoice?"
        ),
        "ground_truth": (
            "10 hours of testing were billed "
            "at 100 dollars per hour, "
            "totaling 1,000 dollars."
        ),
        "doc_id"    : "invoice_inv_2024_001",
        "category"  : "invoice",
        "difficulty": "medium"
    },
    {
        "question": (
            "What does self-attention allow "
            "a model to do?"
        ),
        "ground_truth": (
            "Self-attention allows a model to "
            "relate different positions within "
            "the same sequence to compute "
            "a representation of that sequence, "
            "capturing long-range dependencies "
            "efficiently."
        ),
        "doc_id"    : "attention_is_all_you_need",
        "category"  : "research_paper",
        "difficulty": "medium"
    },
    {
        "question": (
            "What is TechVenture's shareholders "
            "equity?"
        ),
        "ground_truth": (
            "TechVenture's shareholders equity "
            "is 11.56 billion dollars as of "
            "December 31 2023."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "hard"
    },
    {
        "question": (
            "What liability does the developer "
            "have under the contract?"
        ),
        "ground_truth": (
            "The developer's liability is limited "
            "to the total contract value of "
            "120,000 dollars."
        ),
        "doc_id"    : "software_contract_2024",
        "category"  : "contract",
        "difficulty": "medium"
    },
    {
        "question": (
            "How is Llama 2 different from "
            "previous open source models?"
        ),
        "ground_truth": (
            "Llama 2 improves on previous open "
            "source models through increased "
            "context length, grouped query "
            "attention, and extensive safety "
            "fine-tuning using RLHF making it "
            "more helpful and safer."
        ),
        "doc_id"    : "llama2_paper",
        "category"  : "research_paper",
        "difficulty": "medium"
    },
    {
        "question": (
            "What is TechVenture's long term debt?"
        ),
        "ground_truth": (
            "TechVenture has long-term debt of "
            "3.2 billion dollars as listed "
            "in the balance sheet."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "hard"
    },
    {
        "question": (
            "What is the invoice tax rate "
            "and tax amount?"
        ),
        "ground_truth": (
            "The invoice applies a 10 percent "
            "tax rate on the subtotal of "
            "10,400 dollars, resulting in "
            "a tax amount of 1,040 dollars."
        ),
        "doc_id"    : "invoice_inv_2024_001",
        "category"  : "invoice",
        "difficulty": "easy"
    },
    {
        "question": (
            "What are TechVenture's key "
            "growth drivers for 2024?"
        ),
        "ground_truth": (
            "TechVenture's key growth drivers "
            "include AI integration across "
            "products, Asia Pacific expansion, "
            "cybersecurity acquisitions, cloud "
            "migration wave, and an enterprise "
            "contract pipeline of 3.2 billion."
        ),
        "doc_id"    : "financial_report_techventure_2023",
        "category"  : "financial",
        "difficulty": "medium"
    },
]


# ===================================
# Dataset Builder
# ===================================

def build_dataset() -> Dataset:
    """Convert golden pairs to HuggingFace Dataset"""
    return Dataset.from_list(GOLDEN_QA_PAIRS)


def get_by_category(category: str) -> list:
    """Filter questions by category"""
    return [
        q for q in GOLDEN_QA_PAIRS
        if q["category"] == category
    ]


def get_by_difficulty(difficulty: str) -> list:
    """Filter by difficulty level"""
    return [
        q for q in GOLDEN_QA_PAIRS
        if q["difficulty"] == difficulty
    ]


def print_dataset_summary():
    """Print dataset statistics"""
    from collections import Counter

    cats  = Counter(q["category"]   for q in GOLDEN_QA_PAIRS)
    diffs = Counter(q["difficulty"] for q in GOLDEN_QA_PAIRS)

    print("\n" + "=" * 50)
    print("  GOLDEN DATASET SUMMARY")
    print("=" * 50)
    print(f"  Total Q&A pairs: {len(GOLDEN_QA_PAIRS)}")
    print("\n  By Category:")
    for cat, count in cats.most_common():
        print(f"    {cat:<20} : {count}")
    print("\n  By Difficulty:")
    for diff, count in diffs.most_common():
        print(f"    {diff:<20} : {count}")
    print("=" * 50)


if __name__ == "__main__":
    print_dataset_summary()