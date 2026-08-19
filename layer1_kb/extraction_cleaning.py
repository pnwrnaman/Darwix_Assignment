import os
import json
import logging
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader  # Added PDF support

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("extract_and_clean")

BASE_DIR = Path("layer1_kb")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_JSON = PROCESSED_DIR / "records.json"

def clean_html(raw_html):
    """Removes boilerplate navigation and footers from HTML."""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["nav", "footer", "script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def process_files():
    if not DATA_DIR.exists():
        logger.error(f"Data folder not found at {DATA_DIR}! Please create it and add files.")
        return
        
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    records = []
    seen_hashes = set()
    record_counter = 1

    files = list(DATA_DIR.glob("*.*"))
    logger.info(f"Found {len(files)} total files in {DATA_DIR}")

    for file_path in files:
        # Added .pdf to the allowed extensions
        if file_path.suffix.lower() not in [".txt", ".html", ".pdf"]:
            continue

        logger.info(f"Processing: {file_path.name}")
        
        try:
            # Handle PDF extraction
            if file_path.suffix.lower() == ".pdf":
                reader = PdfReader(str(file_path))
                raw_text = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        raw_text.append(text)
                clean_content = "\n".join(raw_text).strip()
            
            # Handle Text and HTML extraction
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
                
                if file_path.suffix.lower() == ".html":
                    clean_content = clean_html(raw_content)
                else:
                    clean_content = raw_content.strip()

            if not clean_content:
                continue

            content_hash = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                logger.warning(f"Duplicate content found in {file_path.name}. Skipping.")
                continue
            
            seen_hashes.add(content_hash)

            records.append({
                "record_id": f"rec_{record_counter:03d}",
                "title": file_path.stem,
                "source": file_path.name,
                "category": "knowledge_base",
                "version": "1.0",
                "has_pii": False, 
                "content": clean_content,
                "extraction_status": "success"
            })
            record_counter += 1

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    if records:
        with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
        logger.info(f"Successfully saved {len(records)} records to {PROCESSED_JSON}")
    else:
        logger.error("No valid text, HTML, or PDF records were extracted!")

if __name__ == "__main__":
    process_files()