"""
DocuSense AI - Gradio Frontend
Production-grade demo UI for recruiters
Features:
- PDF upload and management
- Interactive Q&A with citations
- Real-time system stats
- Pre-loaded example queries
"""

import os
import sys
import json
import time
import httpx
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ===================================
# Config
# ===================================
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000"
)

# Timeout for API calls
TIMEOUT = httpx.Timeout(120.0)

# ===================================
# API Client Functions
# ===================================

def call_api(
    method   : str,
    endpoint : str,
    **kwargs
) -> dict:
    """Generic API caller with error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            if method == "GET":
                resp = client.get(url, **kwargs)
            elif method == "POST":
                resp = client.post(url, **kwargs)
            elif method == "DELETE":
                resp = client.delete(url, **kwargs)
            else:
                return {"error": f"Unknown method: {method}"}

            resp.raise_for_status()
            return resp.json()

    except httpx.ConnectError:
        return {
            "error": (
                "Cannot connect to API server. "
                "Make sure it is running: "
                "python -m uvicorn api.main:app --reload"
            )
        }
    except httpx.TimeoutException:
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        return {"error": str(e)}


# ===================================
# Tab 1: Query Functions
# ===================================

def ask_question(
    question      : str,
    top_k         : int,
    doc_filter    : str,
    history       : list
) -> tuple:
    """
    Send question to API and format response
    Returns: (answer_display, citations_display,
              stats_display, updated_history)
    """
    if not question.strip():
        return (
            "⚠️ Please enter a question.",
            "",
            "",
            history
        )

    # Call API
    payload = {
        "question"      : question,
        "top_k"         : int(top_k),
        "doc_id_filter" : doc_filter if doc_filter != "All Documents" else None
    }

    result = call_api("POST", "/query", json=payload)

    if "error" in result:
        return (
            f"❌ Error: {result['error']}",
            "",
            "",
            history
        )

    # ── Format Answer ────────────────────
    confidence  = result.get("confidence", 0)
    conf_emoji  = (
        "🟢" if confidence >= 0.7 else
        "🟡" if confidence >= 0.4 else
        "🔴"
    )

    answer_text = f"""
{result.get('answer', 'No answer generated')}

---
{conf_emoji} **Confidence:** {confidence:.0%}  
⏱️ **Processing Time:** {result.get('processing_time', 0)}s  
📄 **Documents Used:** {', '.join(result.get('doc_ids_used', []))}  
🔍 **Query Type:** {result.get('query_type', 'hybrid')}  
📦 **Chunks Used:** {result.get('chunks_used', 0)}
    """.strip()

    # ── Format Citations ─────────────────
    citations     = result.get("citations", {})
    citation_list = citations.get("citations", [])

    if citation_list:
        cite_lines = ["### 📚 Sources\n"]
        for cite in citation_list:
            doc   = cite['doc_id'].replace('_', ' ').title()
            ctype = cite['content_type'].replace('_', ' ').title()
            score = cite.get('relevance_score', 0)
            excerpt = cite.get('excerpt', '')[:120]

            cite_lines.append(
                f"**[{cite['citation_number']}] {doc}**  \n"
                f"📄 Page {cite['page_number']} · "
                f"{ctype} · "
                f"Score: {score:.4f}  \n"
                f"*\"{excerpt}...\"*\n"
            )
        citations_display = "\n---\n".join(cite_lines)
    else:
        citations_display = "No citations available."

    # ── Format Retrieval Stats ───────────
    stats       = result.get("retrieval_stats", {})
    stats_display = f"""
### 🔬 Retrieval Statistics

| Metric | Value |
|--------|-------|
| Query Type | {stats.get('query_type', 'N/A')} |
| Semantic Found | {stats.get('semantic_found', 0)} |
| BM25 Found | {stats.get('bm25_found', 0)} |
| After Fusion | {stats.get('after_fusion', 0)} |
| After Rerank | {stats.get('after_rerank', 0)} |
| Has Tables | {result.get('has_tables', False)} |
    """.strip()

    # ── Update Chat History ──────────────
    history = history or []
    history.append((question, result.get('answer', '')))

    return (
        answer_text,
        citations_display,
        stats_display,
        history
    )


def load_example(example_question: str) -> str:
    """Load example question into input box"""
    return example_question


# ===================================
# Tab 2: Document Management
# ===================================

def upload_document(file) -> str:
    """Upload PDF to API"""
    if file is None:
        return "⚠️ Please select a PDF file."

    file_path = Path(file.name)

    if not str(file_path).lower().endswith('.pdf'):
        return "❌ Only PDF files supported."

    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()

        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(
                f"{API_BASE_URL}/ingest/",
                files={
                    "file": (
                        file_path.name,
                        file_content,
                        "application/pdf"
                    )
                }
            )
            response.raise_for_status()
            result = response.json()

        if result.get("status") == "success":
            return f"""
✅ **Document Ingested Successfully!**

📄 **File:** {result.get('file_name')}  
🆔 **Doc ID:** {result.get('doc_id')}  
📦 **Total Chunks:** {result.get('total_chunks')}  
📝 **Text Chunks:** {result.get('text_chunks')}  
📊 **Table Chunks:** {result.get('table_chunks')}  
⏱️ **Time:** {result.get('processing_time')}s  

✅ Document is now searchable!
            """.strip()
        else:
            return f"❌ Ingestion failed: {result}"

    except Exception as e:
        return f"❌ Upload error: {str(e)}"


def list_documents() -> str:
    """Fetch and display document list"""
    result = call_api("GET", "/documents")

    if "error" in result:
        return f"❌ {result['error']}"

    docs  = result.get("documents", [])
    total = result.get("total_documents", 0)
    chunks = result.get("total_chunks", 0)

    if not docs:
        return "📭 No documents ingested yet. Upload a PDF above!"

    lines = [
        f"### 📚 Ingested Documents ({total} total, "
        f"{chunks} total chunks)\n"
    ]

    for doc in docs:
        doc_type = doc.get('doc_type', 'general')
        type_emoji = {
            "research_paper"   : "🔬",
            "financial_report" : "💰",
            "invoice"          : "🧾",
            "contract"         : "📋",
            "general"          : "📄"
        }.get(doc_type, "📄")

        lines.append(
            f"#### {type_emoji} {doc.get('doc_id', 'Unknown')}\n"
            f"- **File:** {doc.get('file_name', 'N/A')}\n"
            f"- **Type:** {doc_type}\n"
            f"- **Pages:** {doc.get('total_pages', 0)}\n"
            f"- **Chunks:** {doc.get('total_chunks', 0)}\n"
            f"- **Size:** {doc.get('file_size_kb', 0):.1f} KB\n"
            f"- **Status:** {doc.get('status', 'unknown')}\n"
        )

    return "\n---\n".join(lines)


def get_doc_choices() -> list:
    """Get document IDs for dropdown"""
    result = call_api("GET", "/documents")
    if "error" in result:
        return ["All Documents"]

    docs    = result.get("documents", [])
    choices = ["All Documents"] + [
        d.get("doc_id", "") for d in docs
    ]
    return choices


# ===================================
# Tab 3: System Health
# ===================================

def get_health() -> str:
    """Get system health status"""
    result = call_api("GET", "/health")

    if "error" in result:
        return f"❌ API not reachable: {result['error']}"

    status  = result.get("status", "unknown")
    emoji   = "🟢" if status == "healthy" else "🔴"

    return f"""
### {emoji} System Health: {status.upper()}

| Component | Status |
|-----------|--------|
| API Server | 🟢 Online |
| Qdrant DB | {'🟢 Connected' if result.get('qdrant_connected') else '🔴 Disconnected'} |
| BM25 Index | {'🟢 Loaded' if result.get('bm25_loaded') else '🟡 Not Loaded'} |
| Total Vectors | {result.get('total_vectors', 0):,} |
| Version | {result.get('version', 'N/A')} |
| Environment | {result.get('environment', 'N/A')} |
    """.strip()


def get_stats() -> str:
    """Get system statistics"""
    result = call_api("GET", "/stats")

    if "error" in result:
        return f"❌ {result['error']}"

    collections = result.get("collections", {})

    col_rows = "\n".join([
        f"| {name.replace('docusense_', '')} "
        f"| {count:,} |"
        for name, count in collections.items()
    ])

    return f"""
### 📊 System Statistics

| Metric | Value |
|--------|-------|
| Total Documents | {result.get('total_documents', 0)} |
| Total Chunks | {result.get('total_chunks', 0):,} |
| Total Vectors | {result.get('total_vectors', 0):,} |
| BM25 Indexed | {result.get('bm25_indexed', 0):,} |
| Environment | {result.get('environment', 'N/A')} |

### 🗄️ Vector Collections

| Collection | Vectors |
|------------|---------|
{col_rows}
    """.strip()


# ===================================
# Example Queries
# ===================================

EXAMPLE_QUERIES = [
    # Research Papers
    [
        "What is the attention mechanism and how does it work?",
        5, "All Documents"
    ],
    [
        "Explain the transformer architecture in simple terms.",
        5, "All Documents"
    ],
    [
        "What makes BERT different from previous language models?",
        5, "All Documents"
    ],
    [
        "What was the main contribution of the RAG paper?",
        5, "All Documents"
    ],
    # Financial
    [
        "What was the total revenue and net income?",
        5, "All Documents"
    ],
    [
        "What are the key financial highlights and growth metrics?",
        5, "All Documents"
    ],
    # Business Docs
    [
        "What are the payment terms and total invoice amount?",
        5, "All Documents"
    ],
    [
        "What are the contract deliverables and timeline?",
        5, "All Documents"
    ],
]


# ===================================
# Build Gradio UI
# ===================================

def create_ui():
    """Create the full Gradio interface"""

    # Custom CSS
    css = """
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .confidence-high { color: #00c851; font-weight: bold; }
    .confidence-med  { color: #ffbb33; font-weight: bold; }
    .confidence-low  { color: #ff4444; font-weight: bold; }
    .answer-box { font-size: 15px; line-height: 1.6; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title = "DocuSense AI",
        theme = gr.themes.Soft(
            primary_hue   = "blue",
            secondary_hue = "indigo"
        ),
        css   = css
    ) as demo:

        # ── Header ───────────────────────────
        gr.HTML("""
        <div class="main-header">
            <h1>🔍 DocuSense AI</h1>
            <p style="font-size:16px; margin:5px 0;">
                Multimodal Document Intelligence Platform
            </p>
            <p style="font-size:13px; opacity:0.8;">
                Hybrid RAG · Semantic + BM25 · 
                Citations · Confidence Scoring
            </p>
        </div>
        """)

        # ── Tabs ─────────────────────────────
        with gr.Tabs():

            # ════════════════════════════════
            # TAB 1: Ask Questions
            # ════════════════════════════════
            with gr.TabItem("💬 Ask Questions", id=0):

                gr.Markdown("""
                ### Ask anything about your documents
                The system searches across all ingested PDFs
                using hybrid retrieval and returns
                cited answers with confidence scores.
                """)

                with gr.Row():

                    # Left column: Input
                    with gr.Column(scale=2):

                        question_input = gr.Textbox(
                            label       = "Your Question",
                            placeholder = (
                                "e.g. What is the attention mechanism? "
                                "| What was total revenue? "
                                "| What are payment terms?"
                            ),
                            lines       = 3,
                            max_lines   = 5
                        )

                        with gr.Row():
                            top_k_slider = gr.Slider(
                                minimum = 1,
                                maximum = 15,
                                value   = 5,
                                step    = 1,
                                label   = "Chunks to Retrieve"
                            )
                            doc_filter = gr.Dropdown(
                                choices = get_doc_choices(),
                                value   = "All Documents",
                                label   = "Filter by Document"
                            )

                        with gr.Row():
                            ask_btn   = gr.Button(
                                "🔍 Ask Question",
                                variant = "primary",
                                scale   = 3
                            )
                            clear_btn = gr.Button(
                                "🗑️ Clear",
                                variant = "secondary",
                                scale   = 1
                            )

                        # Answer output
                        answer_output = gr.Markdown(
                            label = "Answer",
                            value = (
                                "*Your answer will appear here...*"
                            )
                        )

                    # Right column: Citations + Stats
                    with gr.Column(scale=1):

                        citations_output = gr.Markdown(
                            label = "Citations",
                            value = "*Sources will appear here...*"
                        )

                        stats_output = gr.Markdown(
                            label = "Retrieval Stats",
                            value = "*Stats will appear here...*"
                        )

                # Chat history (hidden state)
                history_state = gr.State([])

                # Example queries
                gr.Markdown("### 💡 Example Queries")
                gr.Examples(
                    examples  = EXAMPLE_QUERIES,
                    inputs    = [
                        question_input,
                        top_k_slider,
                        doc_filter
                    ],
                    label     = "Click any example to load it",
                    cache_examples = False
                )

                # Wire up buttons
                ask_btn.click(
                    fn      = ask_question,
                    inputs  = [
                        question_input,
                        top_k_slider,
                        doc_filter,
                        history_state
                    ],
                    outputs = [
                        answer_output,
                        citations_output,
                        stats_output,
                        history_state
                    ],
                    show_progress = True
                )

                question_input.submit(
                    fn      = ask_question,
                    inputs  = [
                        question_input,
                        top_k_slider,
                        doc_filter,
                        history_state
                    ],
                    outputs = [
                        answer_output,
                        citations_output,
                        stats_output,
                        history_state
                    ],
                    show_progress = True
                )

                clear_btn.click(
                    fn      = lambda: (
                        "", "*Your answer will appear here...*",
                        "*Sources will appear here...*",
                        "*Stats will appear here...*", []
                    ),
                    inputs  = [],
                    outputs = [
                        question_input,
                        answer_output,
                        citations_output,
                        stats_output,
                        history_state
                    ]
                )

            # ════════════════════════════════
            # TAB 2: Document Management
            # ════════════════════════════════
            with gr.TabItem("📁 Documents", id=1):

                gr.Markdown("""
                ### Upload & Manage Documents
                Upload any PDF — financial reports,
                research papers, contracts, invoices.
                """)

                with gr.Row():

                    # Upload section
                    with gr.Column(scale=1):
                        gr.Markdown("#### ⬆️ Upload New Document")

                        file_upload = gr.File(
                            label      = "Drop PDF here",
                            file_types = [".pdf"],
                            type       = "filepath"
                        )

                        upload_btn = gr.Button(
                            "📤 Upload & Ingest",
                            variant = "primary"
                        )

                        upload_status = gr.Markdown(
                            value = "*Upload status will appear here*"
                        )

                    # Document list
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📚 Ingested Documents")

                        refresh_btn = gr.Button(
                            "🔄 Refresh List",
                            variant = "secondary"
                        )

                        docs_display = gr.Markdown(
                            value = "*Click Refresh to load documents*"
                        )

                # Wire up
                upload_btn.click(
                    fn      = upload_document,
                    inputs  = [file_upload],
                    outputs = [upload_status]
                )

                refresh_btn.click(
                    fn      = list_documents,
                    inputs  = [],
                    outputs = [docs_display]
                )

                # Auto-load on tab open
                demo.load(
                    fn      = list_documents,
                    inputs  = [],
                    outputs = [docs_display]
                )

            # ════════════════════════════════
            # TAB 3: System Health
            # ════════════════════════════════
            with gr.TabItem("📊 System", id=2):

                gr.Markdown("### System Health & Statistics")

                with gr.Row():
                    health_refresh = gr.Button(
                        "🔄 Refresh Health",
                        variant = "secondary"
                    )
                    stats_refresh = gr.Button(
                        "📊 Refresh Stats",
                        variant = "secondary"
                    )

                with gr.Row():
                    health_display = gr.Markdown(
                        value = "*Click Refresh Health*"
                    )
                    stats_display_main = gr.Markdown(
                        value = "*Click Refresh Stats*"
                    )

                # Architecture info
                gr.Markdown("""
                ---
                ### 🏗️ System Architecture

                ```
                User Query
                    ↓
                Query Router (Zero-shot classification)
                    ↓
                ┌──────────────────────────────┐
                │  Parallel Retrieval          │
                │  ├── Semantic (Qdrant+Gemini)│
                │  └── Keyword (BM25)          │
                └──────────────────────────────┘
                    ↓
                RRF Fusion (Reciprocal Rank Fusion)
                    ↓
                Reranker
                    ↓
                Context Assembly
                    ↓
                Gemini 1.5 Flash (Answer Generation)
                    ↓
                Answer + Citations + Confidence Score
                ```

                ### 🛠️ Tech Stack
                | Component | Technology |
                |-----------|------------|
                | LLM | Google Gemini 1.5 Flash |
                | Embeddings | Gemini embedding-001 |
                | Vector DB | Qdrant Cloud |
                | Keyword Search | BM25 (rank-bm25) |
                | Framework | FastAPI + Gradio |
                | PDF Processing | PyMuPDF + pdfplumber |
                """)

                # Wire up
                health_refresh.click(
                    fn      = get_health,
                    inputs  = [],
                    outputs = [health_display]
                )

                stats_refresh.click(
                    fn      = get_stats,
                    inputs  = [],
                    outputs = [stats_display_main]
                )

                # Auto-load
                demo.load(
                    fn      = get_health,
                    inputs  = [],
                    outputs = [health_display]
                )
                demo.load(
                    fn      = get_stats,
                    inputs  = [],
                    outputs = [stats_display_main]
                )

            # ════════════════════════════════
            # TAB 4: About
            # ════════════════════════════════
            with gr.TabItem("ℹ️ About", id=3):

                gr.Markdown("""
                ## 🔍 DocuSense AI

                **Multimodal Document Intelligence Platform**
                built as a portfolio project demonstrating
                production-grade AI engineering.

                ---

                ### 🎯 What It Does
                - **Ingests** any PDF document
                - **Understands** text, tables, and images
                - **Answers** questions with cited sources
                - **Scores** confidence per answer
                - **Tracks** which retriever found each result

                ---

                ### 🧠 AI Engineering Concepts Demonstrated

                | Concept | Implementation |
                |---------|----------------|
                | RAG | Hybrid retrieval + generation |
                | Embeddings | Gemini embedding-001 (3072d) |
                | Vector Search | Qdrant cosine similarity |
                | Keyword Search | BM25 Okapi |
                | Result Fusion | Reciprocal Rank Fusion |
                | Reranking | Score-based reranker |
                | Chunking | Sentence-aware sliding window |
                | Citations | Page-level attribution |
                | Confidence | Multi-factor scoring |
                | API | FastAPI with OpenAPI docs |

                ---

                ### 📂 Document Types Supported
                - 🔬 Research Papers
                - 💰 Financial Reports
                - 🧾 Invoices
                - 📋 Contracts
                - 📄 General Documents

                ---

                ### 🔗 Links
                - **API Docs:** [localhost:8000/docs](http://localhost:8000/docs)
                - **GitHub:** Your repo link here
                """)

        # ── Footer ───────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:15px;
                    color:#666; font-size:12px;
                    border-top:1px solid #eee;
                    margin-top:20px;">
            DocuSense AI — Built with
            FastAPI · Gradio · Qdrant ·
            Google Gemini · LangChain
        </div>
        """)

    return demo


# ===================================
# Launch
# ===================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("  DOCUSENSE AI - GRADIO FRONTEND")
    print("=" * 55)
    print(f"  API URL: {API_BASE_URL}")
    print("  Make sure API server is running!")
    print("=" * 55 + "\n")

    demo = create_ui()

    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,     # True = public URL
        show_error  = True,
        favicon_path= None
    )