"""
DocuSence AI - Document Fetcher fetches real public documents from:
1. SEC edgar ( financial reports)
2. ArXiv ( research papers )
3. Wikipedia (Tables + text)
All completely Free and legal to use
"""

import os 
import time
import json
import httpx
import fitz # PyMuPDF
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# Setting upd the paths

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
METADATA_FILE = BASE_DIR / "data" / "document_metadata.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#document sources
#sec 10k filings
SEC_DOCUMENTS = [
    {
        "name": "Apple_10K_2023",
        "company": "Apple Inc",
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
        "type": "financial_report",
        "description": "Apple Annual Report 2023"
    },
    {
        "name": "Microsoft_10K_2023",
        "company": "Microsoft corp",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000078901923000039/msft-20230630.htm",
        "type": "financial_report",
        "description": "Microsoft Annual Report 2023"
    }
]

# ArXiv papers -- acadamic papers as pdfs

ARXIV_PAPERS = [
    {
        "name": "attention_is_all_you_need",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762",
        "type": "research_paper",
        "description": "Original Transformer paper"
    },
    {
        "name": "rag_paper",
        "title": "RAG - Retrieval Augmented generation",
        "url": "https://arxiv.org/pdf/2005.11401",
        "type": "research_paper",
        "description": "Original RAG paper by Facebook AI"
    },
    {
        "name": "llama2_paper",
        "title": "Llama 2 Paper",
        "url": "https://arxiv.org/pdf/2307.09288",
        "type": "research_paper",
        "description": "Meta Llama 2 technical report"
    },
    {
        "name": "bert_paper",
        "title": "BERT Paper",
        "url": "https://arxiv.org/pdf/1810.04805",
        "type": "research_paper",
        "description": "BERT: Pre-training of Deep Bidirectioinal Transformers"
    }
]

#sample invoides/contracts (synthetic but realistic)

SAMPLE_DOCS_URLS =[
    {
        "name": "sample_invoice_001",
        "title": "Sample Business Invoice",
        "url": "https://www.w3.org/WAI/WCAG21/Techniques/pdf/img/table-word.pdf",
        "type": "invoice",
        "description": "Sample document with tables"
    }
]

# Fetcher Class

class DocumentFetcher:
    """
    Fetches Real documents from public sources
    Saves them locally with metadata
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "DocuSense-AI-Research/1.0 (educational project)",
            "Accept": "application/pdf,text/html,*/*"
        }
        self.metadata = {}
        self.client = httpx.Client(
            headers=self.headers,
            follow_redirects=True,
            timeout=60.0
        )

    def fetch_pdf(self, url:str, save_path: Path) -> bool:
        """Download a pdf from URL"""

        try:
            logger.info(f"Fetching: {url}")

            response = self.client.get(url)
            response.raise_for_status()

            # Check if it's actually a pdf
            content_type = response.headers.get("content-type", "")
            content = response.content

            #save the file
            with open(save_path, "wb") as f:
                f.write(content)

            logger.success(f"Saved: {save_path.name} ({len(content)/1024:.1f} KB)")
            return True

        except httpx.HTTPError as e:
            logger.error(f"HTTP Error fetching {url}: {e}")
            return False

        except Exception as e:
            logger.error(f"Error fetching {url}:{e}")
            return False

    def validate_pdf(self, pdf_path: Path) -> dict:
        """
        Validate PDF and extract basic info Returns metadata about the document
        """

        try:
            doc = fitz.open(pdf_path)

            # Get Basic info
            page_count = len(doc)

            # get text from first page to validate
            first_page_text = doc[0].get_text()[:200] if page_count > 0 else ""

            # count images
            image_count = sum(
                len(doc[i].get_images()) for i in range(min(5,page_count))
            )

            # pdf metadata
            pdf_meta = doc.metadata

            doc.close()

            return {
                "valid": True,
                "page_count": page_count,
                "image_count_first5": image_count,
                "has_text": len(first_page_text.strip()) > 10,
                "title": pdf_meta.get("title", ""),
                "file_size_kb": pdf_path.stat().st_size / 1024
            }

        except Exception as e:
            logger.error(f"PDF validation failed for {pdf_path}: {e}")
            return {"valid": False, "error": str(e)}

    def create_sample_pdf(self, save_path: Path, doc_type: str):
        """ 
        Creating a realistinc sample PDF when download fails
        Using PyMuPDF to generate real pdfs with text + tables
        """

        doc = fitz.open()

        if doc_type == "invoice":
            self._create_invoice_pdf(doc, save_path)
        elif doc_type == "financial_report":
            self._create_financial_pdf(doc, save_path)
        elif doc_type == "contract":
            self._create_contract_pdf(doc, save_path)
        else:
            self._create_generic_pdf(doc, save_path)

        doc.save(save_path)
        doc.close()
        logger.success(f"Created sample pdf: {save_path.name}")

    def _create_invoice_pdf(self, doc, save_path: Path):
        """Create realistinc invoice pdf"""
        page = doc.new_page()

        content = """
INVOICE

Company: TechCorp solutions Ltd
Invoice Number: INV-2024-001
Date: January 15, 2024
Due Date: February 15, 2024

Bill To:
Acme Corporation
123 Business Street
New Your, Ny 10001

ITEMS:
----------------------------------------------------------
Item               Qty          Unit Price      Total
----------------------------------------------------------
Software Dev       40h          $150/h          $6,000
UI Design          20h          $120/h          $2,400
Testing            10h          $100/h          $1,000
Deployment          5h          $200/h          $1,000
----------------------------------------------------------
                                Subtotal:      $10,400
                                Tax (10%):      $1,040
                                TOTAL:         $11,440
----------------------------------------------------------

Payment Terms: Net 30
Payment Method: Bank Transfer

Bank Details:
Bank: First National Bank
Account: 1234567890
routing: 0210000210

Notes:
Thank you for your business. 
Please include invoice number INV-2024-001 with your payment
        """

        page.insert_text(
            (50, 50),
            content,
            fontsize=11,
            color=(0,0,0)
        )

    def _create_financial_pdf(self, doc, save_path: Path):
        """Create a realistic financial report"""

        # Page 1 : executive summary
        page1 = doc.new_page()

        content1= """
ANNUAL FINANCIAL REPORT 2023
TechVenture Corporation

EXECUTIVE SUMMARY

TechVenture Corporation delivered strong financial results in fiscal year2023,
with total revenue reacing $4.2 billion, representing a 23% increase year-over-year.

Key Financial Highlights:

Revenue:
    Q1 2023: $980 Million ( up 18% YoY)
    Q2 2023: $1.02 Billion (up 21% YoY)
    Q3 2023: $1.08 Billion (up 25% YoY)
    Q4 2023: $1.12 Billion (up 28% YoY)
    Full Year: $4.2 Billion (up 23% YoY)

Profitability:
    Gross Profit: $2.1 Billion (50% margin)
    Operating Income: $840 Million (20% margin)
    Net Income: $674 Million (16% margin)
    EPS: $4.23 (diluted)

Research & Development:
    R&D spend: $630 Million (15% of revenue)
    Patents Filed: 847
    New Products Launched: 12

Geographic Revenue Breakdown:
    North America: 45% ($1.89B)
    Europe: 28%($1.18B)
    Asia Pacific: 20% ($0.84B)
    Rest of Word: 7% ($0.29B)
        """

        page1.insert_text((50,50), content1, fontsize=10)

        # page2 sample balance sheet
        page2 = doc.new_page()

        content2 = """
BALANCE SHEET (as of December 31, 2023)

ASSETS
Current Assets:
    cash and equivalents:       $2,840 Million
    short-term investments:     $1,560 Million
    Accounts receivable:        $890 Million
    Inventory:                  $340 Million
    Prepaid expenses:           $120 Million
    Total Current Assets:       $5,750 Million

Non-Current Assets:
    Property, plant & equipment:    $3,200 Million
    Intangible assets:              $1,800 Million
    Goodwill:                       $4,200 Million
    Long-term investments:          $2,100 Million
    Total Non-Current Assets:       $11,300 Million

TOTAL ASSETS:                       $17,050 Million

LIABILITIES

Current Liabilities:
    Accounts payable:           $620 Million
    Short-term debt:            $400 Million
    Accrued expenses:           $380 Million
    Total Current Liabilities:  $1,400 Million

Non-Current Liabilities:
    Long-term debt:                 $3,200 Million
    Deferred tax liabilities        $890 Million
    Total Non-current Liabilities:  $4,090 Million

TOTAL LIABILITIES:                  $5,490 Million

SHAREHOLDERS EQUITY:                $11,560 Million
TOTAL LIABILITES & EQUITY:          $17,050 Million
        """

        page2.insert_text((50,50), content2, fontsize=10)

        # page 3 inserting some market analysis
        page3 = doc.new_page()

        content3 = """
MARKET ANALYSIS & OUTLOOK 2024

Marker Position:
TechVenture holds the #2 position in enter prise software globally,
with a market share of 18.3%, up from 15.1% in 2022.

Competitive Landscape:
    Competitor A: 24% market share (down 2%)
    TechVenture: 18% market share (up 3%)
    Competitor B: 15% market share (stable)
    others: 43% market share

Product Performance:
    CloudSuite Enterprise:  $1.8B revenue (43% of total)
    DataAnalytics Pro:      $1.1B revenue (26% of total)
    Securityshield:         $840M revenue (20% of total)
    Other Products:         $460M revenue (11% of total)

2024 Guidance:
    Revenue: $5.0-5.2 Billion (19-24% growth)
    Gross Margin: 51-52%
    Operating Margin: 21-22%
    EPS: $5.00-5.20 (diluted)

Key Growth Drivers:
    1. AI integration across product suite
    2. Expansion in Asia Pacific markets
    3. stategic acquisitions in cybersecurity
    4. Cloud migration wave continuing
    5. New enterprise contracts pipeline: $3.2B
        """

        page3.insert_text((50, 50), content3, fontsize=10)


    def _create_contract_pdf(self, doc, save_path: Path):
        """Create realistic contract PDF"""
        page = doc.new_page()
        content = """
SOFTWARE DEVELOPMENT AGREEMENT

This Software Development Agreement ("Agreement") is entered into as of 
January 1, 2024 between:

CLIENT: Acme Corporation, a Delaware corporation
DEVELOPER: TechCorp solution LLC, a New York LLC

1. SERVICES

Developer agrees to provide software development services as
described in Exhibit A ("Services"). The project will be
completed in three phases:

Phase 1 - Requirements & Design (30 days): $25,000
Phase 2 - Development & testing (90 days): $75,000
Phase 3 - Deployment & Training (30 days): $20,000
Total Contract Value: $120,000

2. PAYMENT TERMS
- 30% upfront upon signing: $36,000
- 40% upon Phase 2 completion: $48,000
- 30% upon final delivery: %36,000
- Late Payment Penalty: 1.5% per month

3. INTELLECTUAL PROPERTY
All work product, code, and deliverables created under this
Agreement shall be owned exclusively by Client upon full payment.

4. CONFIDENTIALITY
Developer agrees to maintain strict confidentiality of all
Client information, trade secrets, and business data.

5. TERM and TERMINATION
This Agreement begin January 1, 2024 and ends June 30, 2024.
either party may terminate with 30 days written notice.

6. LIABILILITY
Developer liability limited to total contract value of $120,000.

signed:
Client:________________________ Date: January 1, 2024
Developer:________________________Date: January 1, 2024
        """

        page.insert_text((50, 50), content, fontsize=10)

    def _create_generic_pdf(self, doc, save_path: Path):
        """Creating a generic document"""
        page = doc.new_page()
        page.insert_text(
            (50,50),
            "DocuSense AI - Sample Document\n\n This is a sample document.",
            fontsize=12
        )

    def fetch_arxiv_papers(self):
        """Fetch ArXiv research papers"""
        logger.info("=" * 50)
        logger.info("Fetching arxiv papers")
        logger.info("=" * 50)

        results = []

        for paper in tqdm(ARXIV_PAPERS, desc="Downloading papers"):
            save_path = RAW_DIR / f"{paper['name']}.pdf"

            if save_path.exists():
                logger.info(f"Already exists: {paper['name']}")
                meta = self.validate_pdf(save_path)
                meta.update(paper)
                results.append(meta)
                continue

            success = self.fetch_pdf(paper["url"], save_path)

            if success:
                meta = self.validate_pdf(save_path)
            else:
                # create sample if download fails
                logger.warning(f"Download failed creating a sameple for {paper['name']}")
                self.create_sample_pdf(save_path, "research_paper")
                meta = self.validate_pdf(save_path)

            meta.update(paper)
            results.append(meta)

            time.sleep(2)

        return results

    def fetch_sample_business_docs(self):
        """Create sample business documents"""
        logger.info("=" * 50)
        logger.info("Creating sample business documents")
        logger.info("=" * 50)

        results = []

        sample_docs = [
            ("financial_report_techventure_2023", "financial_report"),
            ("invoice_inv_2024_001", "invoice"),
            ("software_contract_2024", "contract"),
            ("financial_report_techventure_2022", "financial_report"),
        ]

        for doc_name, doc_type in tqdm(sample_docs, desc="Creating samples"):
            save_path = RAW_DIR / f"{doc_name}.pdf"

            if save_path.exists():
                logger.info(f"Already exist: {doc_name}")
            else:
                self.create_sample_pdf(save_path, doc_type)

            meta = self.validate_pdf(save_path)
            meta.update({
                "name": doc_name,
                "type": doc_type,
                "description": f"Sample {doc_type} document"
            })
            results.append(meta)

        return results

    def save_metadata(self, all_results: list):
        """Save document metadata to JSON"""
        metadata = {}
        for result in all_results:
            name = result.get("name", "unknows")
            metadata[name] = result

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.success(f"Metadata save to {METADATA_FILE}")
        return metadata

    def print_summary(self, metadata: dict):
        """Print summary of fetched documents"""
        print("\n" + "=" * 60)
        print("DOCUSENSE AI - Document fetch summary")
        print("="*60)

        total_pages = 0
        total_size = 0

        for name, meta in metadata.items():
            status = "ok" if meta.get("valid") else "not ok"
            pages = meta.get("page_count", 0)
            size = meta.get("file_size_kb", 0)
            doc_type = meta.get("type", "unknown")

            total_pages += pages
            total_size += size

            print(f"\n {status}{name}")
            print(f" type {doc_type}")
            print(f" pages {pages}")
            print(f" size {size:.1f} KB")

        print("\n" + "-"*60)
        print(f" total documents: {len(metadata)}")
        print(f" total pages: {total_pages}")
        print(f" total size {total_size/1024:.2f} MB")
        print("\n" + "-"*60)
        print(" documents ready for ingestion pipeline!")
        print(f" Location: {RAW_DIR}")
        print("="*60)


# main function
def main():
    logger.info("Starting docusense ai document fetcher")

    fetcher = DocumentFetcher()
    all_results =[]

    arxiv_results = fetcher.fetch_arxiv_papers()
    all_results.extend(arxiv_results)

    business_results = fetcher.fetch_sample_business_docs()
    all_results.extend(business_results)

    metadata = fetcher.save_metadata(all_results)

    fetcher.print_summary(metadata)

    return metadata

if __name__ == "__main__":
    main()
