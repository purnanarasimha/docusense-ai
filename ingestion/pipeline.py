"""Main ingestion pipe line Orchestrates: Parse extract caption chunk save"""

import json
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from ingestion.pdf_parser import BatchPDFParser
from ingestion.table_extractor import BatchTableExtractor
from ingestion.image_captioner import ImageCaptioner
from ingestion.chunk_strategy import SmartChunker

load_dotenv()

# paths

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PARSED_DIR = PROCESSED_DIR / "parsed"
TABLES_DIR = PROCESSED_DIR / "tables"
IMAGES_DIR = PROCESSED_DIR / "images"
CHUNKS_DIR = PROCESSED_DIR / "chunks"

class IngestionPipeline:
    """ Full document ingestion pipe line pdf -> parse -> tables -> images -> chunks"""

    def __init__(self):
        self.pdf_parser = BatchPDFParser()
        self.table_extractor = BatchTableExtractor()
        self.image_captioner = ImageCaptioner()
        self.chunker = SmartChunker()

    def run(self, input_dir: Path = RAW_DIR, skip_images: bool = False) -> dict:
        """run full ingestion pipe lien on all pdfs keep skip images: True for testing to save api calls"""

        logger.info("="*55)
        logger.info(" docusense ai - ingestion pipe line started")
        logger.info("="*55)

        for d in [PARSED_DIR, TABLES_DIR, IMAGES_DIR, CHUNKS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error("No pdfs found in {inpurt_dir}")
            return {}
        logger.info(f"Found {len(pdf_files)} pdfs in process")

        all_chunks = []
        results = {}

        for pdf_path in pdf_files:
            doc_id = pdf_path.stem
            logger.info(f"\nProcessing : {pdf_path.name}")

            doc_result = {
                "doc_id": doc_id,
                "text_chunks": 0,
                "table_chunks": 0,
                "image_chunks": 0,
                "total_chunks": 0
            }

            try:
                # step for pdf parse text
                logger.info(" [1/4] Parsing text...")
                parsed_results = self.pdf_parser.parse_directory(input_dir=input_dir, output_dir=PARSED_DIR)

                parsed_path = PARSED_DIR / f"{doc_id}_parsed.json"

                # step to extract tables
                logger.info(" [2/4] extracting tables...")
                tables = self.table_extractor.extractor.extract_from_pdf(pdf_path)
                tables_path = None

                if tables:
                    tables_path = self.table_extractor.extractor.save_tables(tables, TABLES_DIR, doc_id)

                # step to extract and caption images
                images_path = None
                if not skip_images:
                    logger.info(" [3/4] captioning images")
                    images = self.image_captioner.extract_and_caption(pdf_path,IMAGES_DIR)
                    if images:
                        images_path = self.image_captioner.save_captions(images,IMAGES_DIR,doc_id)

                else:
                    logger.info(" [3/4] skip images is set true")

                # step to create chunks
                logger.info(" [4/4] creating the chunks")
                if parsed_path.exists():
                    chunks = self.chunker.chunk_document(parsed_path=parsed_path,tables_path=tables_path,images_path=images_path)
                    # save chunks
                    self.chunker.save_chunks(chunks,CHUNKS_DIR,doc_id)
                    all_chunks.extend(chunks)

                    #count by type
                    doc_result["text_chunks"] = sum(1 for c in chunks if c.content_type == "text")
                    doc_result["table_chunks"] = sum(1 for c in chunks if c.content_type == "table")
                    doc_result["image_chunks"] = sum(1 for c in chunks if c.content_type == "image_caption")
                    doc_result["total_chunks"] = len(chunks)

                results[doc_id] = doc_result
                logger.success(f" done: {doc_result['total_chunks']} chunks")

            except Exception as e:
                logger.error(f" failed: {pdf_path.name}: error {e}")
                results[doc_id] = {"error": str(e)}

        self._save_master_index(all_chunks)

        self._print_final_summary(results, all_chunks)

        return {
            "results": results,
            "total_chunks": len(all_chunks),
            "chunks": all_chunks
        }

    def run_single(self, pdf_path: Path, skip_images: bool = True) -> list:
        """process a single pdf file"""
        doc_id = pdf_path.stem
        for d in [PARSED_DIR,TABLES_DIR,IMAGES_DIR,CHUNKS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        from ingestion.pdf_parser import PDFParser
        parser = PDFParser()
        doc_content = parser.parse(pdf_path)
        parsed_path = PARSED_DIR / f"{doc_id}_parsed.json"
        parser.save_parsed(doc_content, PARSED_DIR)

        # tables 
        tables = self.table_extractor.extractor.extract_from_pdf(pdf_path)
        tables_path = None
        if tables:
            tables_path = self.table_extractor.extractor.save_tables(tables, TABLES_DIR, doc_id)

        # images
        images_path = None
        if not skip_images:
            images = self.image_captioner.extract_and_caption(pdf_path, IMAGES_DIR)
            if images:
                images_path = self.image_captioner.save_captions(images, IMAGES_DIR, doc_id)
        #chunks
        chunks = self.chunker.chunk_document(parsed_path=parsed_path,tables_path=tables_path,images_path=images_path)
        self.chunker.save_chunks(chunks, CHUNKS_DIR, doc_id)
        return chunks

    def _save_master_index(self, all_chunks: list):
        """save master index of all chunks"""
        index = []
        for chunk in all_chunks:
            from dataclasses import asdict
            index.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content_type": chunk.content_type,
                "page_number": chunk.page_number,
                "word_count": chunk.word_count,
            })

        index_path = PROCESSED_DIR / "master_index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        logger.info(f"Master index saved: {len(index)} chunks total")

    def _print_final_summary(self, results: dict, all_chunks: list):
        print("\n" + "="*60)
        print(" INGESTIONG PIPELINE COMPLETE")
        print("="*60)

        for doc_id, result in results.items():
            if "error" in result:
                print(f"\n {doc_id}: {result['error']}")
            else:
                print(f"\n {doc_id}")
                print(f" text chunks: {result['text_chunks']} ")
                print(f" table chunks: {result['table_chunks']} ")
                print(f" image chunks: {result['image_chunks']} ")
                print(f" Total: {result['total_chunks']}")

        print("\n" + "-"*60)
        total = len(all_chunks)
        text_total = sum(1 for c in all_chunks if c.content_type == "text")
        table_total = sum(1 for c in all_chunks if c.content_type == "table")
        image_total = sum(1 for c in all_chunks if c.content_type == "image_caption")

        print(f"    total chunks: {total}")
        print(f"    text: {text_total}")
        print(f"    tables: {table_total}")
        print(f"    images: {image_total}")
        print("=" * 60)
        print("ready for embeddings")
        print("="*60)

# run the pipe line
if __name__ == "__main__":
    pipeline = IngestionPipeline()

    result = pipeline.run(skip_images=True)