"""
Extract images from pdfs and generates captions uses google gemini vision for captioning
"""

import os
import json
import base64
import fitz #PyMuPDF
import google.generativeai as genai
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

# configure gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# data models

@dataclass
class ExtractedImage:
    """Represents an extracted image with caption"""
    image_id: str
    doc_id: str
    page_number: int
    image_index: int
    caption: str
    width: int
    height: int
    image_path: str

# image captioner class

class ImageCaptioner:
    """Extract image and generate caption using gemini"""

    def __init__(self):
        self.model = genai.GenerativeModel("gemini-3.5-flash-lite")
        self.min_size = 100 # to skip tiny images icons and bullets
        self.max_images_per_doc = 20

    def extract_and_caption(self, pdf_path: Path, images_dir: Path) -> list:
        """Extract images from pdf and generated captions"""

        logger.info(f"processing images from: {pdf_path.name}")
        doc_id = pdf_path.stem

        doc_images_dir = images_dir / doc_id
        doc_images_dir.mkdir(parents=True, exist_ok=True)

        extracted_images = []

        try:
            doc = fitz.open(pdf_path)
            image_count = 0

            for page_num in range(len(doc)):
                if image_count >= self.max_images_per_doc:
                    logger.info(f"Reached max images limit ({self.max_images_per_doc})")
                    break

                page = doc[page_num]
                images = page.get_images()

                for img_idx, img_info in enumerate(images):
                    if image_count >= self.max_images_per_doc:
                        break

                    try:
                        # Extract image
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # open with PIL to check size
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        width, height = pil_image.size

                        if width < self.min_size or height < self.min_size:
                            continue

                        # save image
                        image_filename = (f"{doc_id}_p{page_num+1}_img{img_idx}.{image_ext}")

                        image_path = doc_images_dir / image_filename

                        with open(image_path, "wb") as f:
                            f.write(image_bytes)

                        # Generate caption
                        caption = self._generate_caption(pil_image,doc_id, page_num + 1)

                        extracted_image = ExtractedImage(
                            image_id = f"{doc_id}_p{page_num+1}_img{img_idx}.{image_ext}",
                            doc_id=doc_id,
                            page_number=page_num+1,
                            image_index=img_idx,
                            caption=caption,
                            width=width,
                            height=height,
                            image_path=str(image_path)
                        )

                        extracted_images.append(extracted_image)
                        image_count += 1

                        logger.info(f"Image {image_count}: page {page_num + 1}, {width}x{height}px")

                    except Exception as e:
                        logger.warning(f"Falied to process image {img_idx} on page {page_num+1}: {e}")
                        continue

            doc.close()

        except Exception as e:
            logger.error(f"Image extraction failed for {pdf_path.name}:{e}")

        logger.success(f"Extracted {len(extracted_images)} images from {pdf_path.name}")

        return extracted_images

    def _generate_caption(self, image: Image.Image, doc_id: str, page_num: int) -> str:
        """generate descriptive caption using gemini"""
        try:
            prompt = """Analyze the given image from a document and provide the following:
            1. what is the type of image (chart,graph,diagram,photo,table screenshot)?
            2. What is the main content or information shown?
            3. Any specific numbers, labels, or key data points visible?
            
            Provide a concise but information caption in 2 or 3 sentences which will help someone to understand the image content with out seeing it. More focus on information value. """

            response = self.model.generate_content([prompt, image])
            caption = response.text.strip()

            return caption

        except Exception as e:
            logger.warning(f"caption generation failed: {e}")
            return f"Image from document {doc_id}, page {page_num}. Content could not be automatically captioned"

    def save_captions(self, images: list, ouput_dir: Path, doc_id: str) -> Path:
        """Save image captions to json"""
        ouput_dir.mkdir(parents=True, exist_ok=True)
        ouput_path = ouput_dir / f"{doc_id}_images.json"
        images_data = [asdict(img) for img in images]
        with open(ouput_path, "w", encoding="utf-8") as f:
            json.dump(images_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(images)} image records")
        return ouput_path

    