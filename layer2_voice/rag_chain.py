import os
import logging
import asyncio
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_chain")

# 1. Environment Variables
load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", 
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 4))

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "layer1_kb", "chroma_db")
)

# 2. Local HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

# 3. Vector Store + MMR Retriever
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_K * 3, "lambda_mult": 0.5}
)

# 4. Local Ollama Model
llm = ChatOllama(model=LLM_MODEL, temperature=0.0)

# 5. Prompts
NO_INFO_RESPONSE = "I'm sorry, but that information is unavailable in our knowledge base."

SYSTEM_TEXT = """You are a professional AI voice agent assistant.
Answer ONLY using the context provided. Do not invent, assume, or infer facts not explicitly present in the context.
If the answer is not in the context, respond exactly with:
"I'm sorry, but that information is unavailable in our knowledge base."
Never mention that you are an AI or that you were given context."""

HUMAN_TEXT = """Context:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_TEXT),
    HumanMessagePromptTemplate.from_template(HUMAN_TEXT),
])

def format_docs(docs):
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(
            f"[record_id: {m.get('record_id', 'N/A')} | source: {m.get('source', 'Unknown')} "
            f"| category: {m.get('category', 'General')}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)

def build_citations(docs):
    citations = []
    for doc in docs:
        m = doc.metadata
        citations.append({
            "record_id": m.get("record_id"),
            "chunk_id": m.get("chunk_id"),
            "source": m.get("source"),
            "category": m.get("category"),
            "snippet": doc.page_content[:200],
        })
    return citations

async def aquery(question: str) -> dict:
    docs = await retriever.ainvoke(question)

    if not docs:
        logger.info("No relevant chunks retrieved for query: %s", question)
        return {
            "question": question,
            "answer": NO_INFO_RESPONSE,
            "citations": [],
            "grounded": False,
        }

    context_str = format_docs(docs)
    chain = prompt | llm | StrOutputParser()
    
    answer = await chain.ainvoke({"context": context_str, "question": question})
    is_grounded = NO_INFO_RESPONSE.lower() not in answer.lower()

    return {
        "question": question,
        "answer": answer,
        "citations": build_citations(docs) if is_grounded else [],
        "grounded": is_grounded,
    }



if __name__ == "__main__":
    test_q = "What are the health insurance qualification requirements?"
    print(f"\n[+] Running Standalone RAG Chain Smoke Test...")
    print(f"[+] Query: '{test_q}'\n")

    response = asyncio.run(aquery(test_q))

    print("================ RAG OUTPUT ================")
    print(f"Answer:   {response['answer']}")
    print(f"Grounded: {response['grounded']}")
    print(f"Citations Found: {len(response['citations'])}")
    for idx, c in enumerate(response['citations'], 1):
        print(f"   {idx}. [Source: {c['source']}] {c['snippet'][:80]}...")
    print("============================================\n")