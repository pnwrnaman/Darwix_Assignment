import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("test_retrieval")

CHROMA_DIR = Path("layer1_kb/chroma_db")


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

TOP_K = int(os.getenv("TEST_TOP_K", "3"))


TEST_QUERIES = [
    {
        "query": "What is the health insurance grace period?",
        "expect_keywords": ["grace period", "30-day"],
        "expect_category": "knowledge_base",
    },
    {
        "query": "What are the qualifications for a business loan?",
        "expect_keywords": ["operational", "12 months", "100,000"],
        "expect_category": "knowledge_base",
    },
    {
        "query": "What is the waiting period for BCA Life health coverage?",
        "expect_keywords": ["masa tunggu", "bca", "asuransi"], # Tests if English query fetches Indonesian text!
        "expect_category": "knowledge_base",
    }
]


def load_vector_db() -> Chroma:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"{CHROMA_DIR} not found. Run chunk_and_embed.py first!"
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)


def check_result(query_spec: dict, results: list[tuple]) -> bool:
    """Loose grounding check: do the expected keywords show up anywhere
    in the top-k retrieved content, or does the top hit match the
    expected category? Either signal counts as a pass."""
    if not results:
        return False

    combined_text = " ".join(doc.page_content.lower() for doc, _ in results)
    keyword_hit = any(kw.lower() in combined_text for kw in query_spec.get("expect_keywords", []))

    top_category = results[0][0].metadata.get("category", "")
    category_hit = query_spec.get("expect_category") == top_category

    return keyword_hit or category_hit


def run_query(vector_db: Chroma, query_spec: dict) -> bool:
    query_text = query_spec["query"]
    print(f"\n=== QUERY: '{query_text}' ===")

    try:
        results = vector_db.similarity_search_with_relevance_scores(query_text, k=TOP_K)
    except Exception:
        logger.exception("Retrieval failed for query: %s", query_text)
        return False

    if not results:
        print("  (no results returned)")
        return False

    for doc, score in results:
        print(f"Record ID: {doc.metadata.get('record_id')}")
        print(f"Chunk ID:  {doc.metadata.get('chunk_id')}")
        print(f"Source:    {doc.metadata.get('source')}")
        print(f"Category:  {doc.metadata.get('category')}")
        print(f"Score:     {score:.4f}")
        print(f"Content:   {doc.page_content[:150]}...\n")

    passed = check_result(query_spec, results)
    print(f"  -> {'PASS' if passed else 'FAIL'} (grounding check)")
    return passed


def main():
    try:
        vector_db = load_vector_db()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    results = [run_query(vector_db, q) for q in TEST_QUERIES]

    passed = sum(results)
    total = len(results)
    print(f"\n=== SUMMARY: {passed}/{total} queries passed grounding check ===")
    if passed < total:
        print("Review FAILed queries above — either the KB is missing that "
              "content, or expect_keywords/expect_category need updating "
              "to match your actual data/ documents.")


if __name__ == "__main__":
    main()