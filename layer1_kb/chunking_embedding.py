import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("chunk_and_embed")

PROCESSED_JSON = Path("layer1_kb/processed/records.json")
CHROMA_DIR = Path("layer1_kb/chroma_db")


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP", "40"))


def load_records() -> list[dict]:
    if not PROCESSED_JSON.exists():
        raise FileNotFoundError(
            f"{PROCESSED_JSON} not found. Run extract_and_clean.py first!"
        )
    with PROCESSED_JSON.open("r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d records from %s", len(records), PROCESSED_JSON)
    return records


def records_to_documents(records: list[dict]) -> list[Document]:
    documents = []
    skipped = 0
    for rec in records:
        if rec.get("extraction_status") == "failed" or not rec.get("content", "").strip():
            skipped += 1
            continue

        doc = Document(
            page_content=rec["content"],
            metadata={
                "record_id": rec["record_id"],
                "title": rec.get("title", ""),
                "source": rec.get("source", ""),
                "category": rec.get("category", ""),
                "version": rec.get("version", ""),
                "has_pii": rec.get("has_pii", False),
            },
        )
        documents.append(doc)

    if skipped:
        logger.warning("Skipped %d records (failed extraction or empty content)", skipped)
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    # Token counting runs locally and is 100% free
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    counts_per_record: dict[str, int] = defaultdict(int)

    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        record_id = doc.metadata["record_id"]

        for chunk in doc_chunks:
            counts_per_record[record_id] += 1
            idx = counts_per_record[record_id]
            chunk.metadata["chunk_id"] = f"{record_id}_c{idx:03d}"
            chunk.metadata["chunk_index_in_doc"] = idx
            chunk.metadata["chunk_count_in_doc"] = None 

        all_chunks.extend(doc_chunks)

        total = counts_per_record[record_id]
        for chunk in doc_chunks:
            chunk.metadata["chunk_count_in_doc"] = total

    logger.info(
        "Generated %d chunks from %d documents.",
        len(all_chunks), len(documents)
    )
    return all_chunks


def build_vector_store():
    try:
        records = load_records()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    documents = records_to_documents(records)
    if not documents:
        logger.error("No usable documents after filtering. Nothing to embed.")
        return

    chunks = chunk_documents(documents)
    if not chunks:
        logger.error("Chunking produced zero chunks. Aborting before embedding.")
        return

    chunk_ids = [c.metadata["chunk_id"] for c in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        logger.error("Duplicate chunk_ids detected — aborting.")
        return

    
    logger.info(f"Downloading/Loading local model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    try:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            ids=chunk_ids,
            persist_directory=str(CHROMA_DIR),
        )
    except Exception:
        logger.exception("Embedding/indexing failed.")
        return

    try:
        count = vector_db._collection.count()
    except Exception:
        count = "unknown"

    logger.info("Successfully indexed vector database at %s", CHROMA_DIR)
    logger.info("Chunks generated: %d | Chunks in collection: %s", len(chunks), count)


if __name__ == "__main__":
    build_vector_store()