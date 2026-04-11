from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings(
    "ignore",
    message="The 'validate_default' attribute with value True was provided to the `Field\\(\\)` function.*",
)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from graphrag_pipeline import (
    graph_checkpoint_exists,
    hybrid_query,
    load_graph_checkpoint,
    load_or_build_communities,
    load_or_build_community_summaries,
)
from llm_utils import _maybe_build_openai_llm
from safety_shield import US_CRISIS_RESOURCES


load_dotenv(Path(".env"), override=False)


DEFAULT_GRAPH_CHECKPOINT_CANDIDATES = (
    "./checkpoints/clinical_dsm_merged",
    "./checkpoints/clinical_graph_clean",
)


def _default_graph_checkpoint_dir() -> str:
    configured = os.getenv("GRAPH_CHECKPOINT_DIR")
    if configured:
        return configured

    for candidate in DEFAULT_GRAPH_CHECKPOINT_CANDIDATES:
        if graph_checkpoint_exists(candidate):
            return candidate
    return DEFAULT_GRAPH_CHECKPOINT_CANDIDATES[0]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to answer with GraphRAG.")
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier to let follow-up symptom queries refine earlier ones.",
    )
    answer_with_llm: bool = Field(
        default=False,
        description="If true, use the configured answer LLM for synthesis.",
    )


class QueryResponse(BaseModel):
    source_chunks: list[dict[str, str]]
    answer: str
    query_type: str
    citations: list[str]
    is_crisis: bool
    safety_resources: list[dict[str, str]]
    matched_safety_indicators: list[str]
    out_of_scope: bool
    communities_used: list[str]


class GraphRAGService:
    def __init__(self) -> None:
        self.graph_checkpoint_dir = _default_graph_checkpoint_dir()
        self.community_checkpoint_dir = os.getenv("COMMUNITY_CHECKPOINT_DIR", self.graph_checkpoint_dir)
        self.session_questions: dict[str, list[str]] = {}

        self.artifacts, self.graph_meta = load_graph_checkpoint(self.graph_checkpoint_dir)
        self.communities = load_or_build_communities(
            self.artifacts,
            checkpoint_dir=self.community_checkpoint_dir,
            force_rebuild=False,
        )
        self.summaries = load_or_build_community_summaries(
            self.communities,
            self.artifacts,
            checkpoint_dir=self.community_checkpoint_dir,
            llm_client=None,
            force_rebuild=False,
        )
        self.answer_llm = _maybe_build_openai_llm("PIPELINE_ANSWER", "gpt-4o-mini", 400)

    def query(
        self,
        question: str,
        *,
        answer_with_llm: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        history: list[str] = []
        normalized_session_id = (session_id or "").strip()
        if normalized_session_id:
            history = list(self.session_questions.get(normalized_session_id, []))

        result = hybrid_query(
            question,
            self.artifacts,
            community_summaries=self.summaries,
            llm_client=self.answer_llm,
            answer_with_llm=answer_with_llm and self.answer_llm is not None,
            conversation_context=history,
        )
        if normalized_session_id:
            updated_history = [*history, question][-6:]
            self.session_questions[normalized_session_id] = updated_history
        return result


@lru_cache(maxsize=1)
def get_service() -> GraphRAGService:
    return GraphRAGService()


app = FastAPI(
    title="Clinical GraphRAG API",
    version="1.0.0",
    description="FastAPI wrapper around the repaired clinical GraphRAG pipeline.",
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "Clinical GraphRAG API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "meta": "/meta",
        "query": "/query",
        "safety_resources": "/safety-resources",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> JSONResponse:
    return JSONResponse(status_code=204, content=None)


@app.get("/health")
def health() -> dict[str, Any]:
    service = get_service()
    return {
        "status": "ok",
        "graph_checkpoint_dir": service.graph_checkpoint_dir,
        "community_checkpoint_dir": service.community_checkpoint_dir,
        "entity_count": service.artifacts.entity_count,
        "relation_count": service.artifacts.relation_count,
        "community_count": len(service.communities),
        "summary_count": len(service.summaries),
    }


@app.get("/meta")
def meta() -> dict[str, Any]:
    service = get_service()
    return {
        "graph_meta": service.graph_meta,
        "entity_count": service.artifacts.entity_count,
        "relation_count": service.artifacts.relation_count,
        "community_count": len(service.communities),
        "summary_count": len(service.summaries),
        "answer_llm_configured": service.answer_llm is not None,
    }


@app.get("/safety-resources")
def safety_resources() -> dict[str, Any]:
    return {"resources": US_CRISIS_RESOURCES}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = get_service().query(
            request.question,
            answer_with_llm=request.answer_with_llm,
            session_id=request.session_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GraphRAG query failed: {exc}") from exc

    return QueryResponse(
        source_chunks=[
            {
                "chunk_id": str(item.get("chunk_id", "")),
                "citation": str(item.get("citation", "")),
                "text": str(item.get("text", "")),
                "section_title": str(item.get("section_title", "")),
                "context_tag": str(item.get("context_tag", "")),
                "source_label": str(item.get("source_label", "")),
                "file_path": str(item.get("file_path", "")),
                "header_path": str(item.get("header_path", "")),
            }
            for item in result.get("source_chunks", [])
        ],
        answer=result["answer"],
        query_type=result["query_type"],
        citations=result["citations"],
        is_crisis=result["is_crisis"],
        safety_resources=result.get("safety_resources", []),
        matched_safety_indicators=result.get("matched_safety_indicators", []),
        out_of_scope=result["out_of_scope"],
        communities_used=[str(item) for item in result["communities_used"]],
    )
