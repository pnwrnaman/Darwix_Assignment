import sys
import os
import logging
import httpx
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("layer2_api")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app_state = {"kb_ready": False, "startup_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global aquery, retriever, LLM_PROVIDER, LLM_MODEL
        from layer2_voice.rag_chain import aquery, retriever, LLM_PROVIDER, LLM_MODEL

        count = retriever.vectorstore._collection.count()
        if count == 0:
            logger.warning("ChromaDB collection is empty — check Layer 1 ingestion.")
        else:
            logger.info("Connected to ChromaDB with %d chunks.", count)

        app_state["kb_ready"] = True
        logger.info("Layer 2 startup complete. Provider=%s Model=%s", LLM_PROVIDER, LLM_MODEL)
    except Exception as e:
        app_state["startup_error"] = str(e)
        logger.error("Layer 2 startup failed: %s", e, exc_info=True)
   
    yield
    logger.info("Layer 2 shutting down.")


app = FastAPI(
    title="Darwix Layer 2 - Local RAG API (Ollama)",
    description="100% Free and Local Asynchronous RAG Backend",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        example="What are the eligibility rules for health insurance qualification?",
    )

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be blank or whitespace-only")
        return v


class CitationSchema(BaseModel):
    record_id: Optional[str] = "N/A"
    chunk_id: Optional[str] = "N/A"
    source: Optional[str] = "Unknown"
    category: Optional[str] = "General"
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationSchema]
    grounded: bool



@app.get("/")
async def root():
    return {
        "service": "Darwix Layer 2 RAG API",
        "docs": "/docs",
        "health": "/health",
        "query": "/query (POST)",
    }


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Reports actual dependency state, not a static string."""
    if app_state["startup_error"]:
        return {
            "status": "unhealthy",
            "layer": 2,
            "error": app_state["startup_error"],
        }

    try:
        from layer2_voice.rag_chain import LLM_PROVIDER, LLM_MODEL, retriever
        chunk_count = retriever.vectorstore._collection.count()
        return {
            "status": "healthy" if app_state["kb_ready"] else "degraded",
            "layer": 2,
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "kb_chunk_count": chunk_count,
        }
    except Exception as e:
        return {"status": "unhealthy", "layer": 2, "error": str(e)}


@app.post("/query", status_code=status.HTTP_200_OK)
async def handle_query(request: Request):
    if not app_state["kb_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base or LLM backend is not ready. Check /health.",
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

   
    is_vapi = "message" in body and "toolCallList" in body.get("message", {})

    if is_vapi:
        tool_calls = body["message"]["toolCallList"]
        if not tool_calls:
            return {"results": [{"toolCallId": "", "result": "No tool calls provided."}]}
        
        tool_call = tool_calls[0]
        tool_call_id = tool_call.get("id")
        arguments = tool_call.get("function", {}).get("arguments", {})
        
        
        question = arguments.get("question") or arguments.get("query") or arguments.get("text") or ""
        
        if not question or len(question.strip()) < 2:
            return {
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "result": "I didn't catch that question clearly. Could you please repeat it?"
                    }
                ]
            }

        try:
           
            rag_result = await aquery(question.strip())
            
           
            if isinstance(rag_result, dict):
                answer_text = rag_result.get("answer", str(rag_result))
            else:
                answer_text = getattr(rag_result, "answer", str(rag_result))
            
            
            return {
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "result": answer_text
                    }
                ]
            }
        except Exception as e:
            logger.error("Error processing Vapi RAG query: %s", e, exc_info=True)
            return {
                "results": [
                    {
                        "toolCallId": tool_call_id,
                        "result": "I am having trouble accessing the knowledge base right now. Please try again."
                    }
                ]
            }

    else:
        
        question = body.get("question")
        if not question or not isinstance(question, str) or len(question.strip()) < 2:
            raise HTTPException(status_code=422, detail="Invalid question format. 'question' field is required.")
        
        try:
            result = await aquery(question.strip())
            return result
        except (ConnectionError, httpx.ConnectError) as e:
            logger.error("LLM backend unreachable: %s", e, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The local AI model service is currently unreachable. Please verify Ollama is running.",
            )
        except Exception as e:
            logger.error("Error processing RAG query: %s", e, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while generating the response.",
            )