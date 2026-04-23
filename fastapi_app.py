from __future__ import annotations

import os
import time
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
    GraphArtifacts,
    graph_checkpoint_exists,
    hybrid_query,
    load_or_build_communities,
    load_or_build_community_summaries,
)
from llm_utils import _maybe_build_openai_llm
from neo4j_helpers import (
    Neo4jGraphQueryBackend,
    create_driver,
    get_neo4j_graph_stats,
    restore_from_neo4j,
    test_connection,
)
from safety_shield import US_CRISIS_RESOURCES


load_dotenv(Path(".env"), override=True)


DEFAULT_GRAPH_CHECKPOINT_CANDIDATES = (
    "./checkpoints/clinical_dsm_merged",
    "./checkpoints/clinical_graph_clean",
)


def _default_graph_sync_source_dir() -> str:
    configured = os.getenv("GRAPH_CHECKPOINT_DIR") or os.getenv("GRAPH_SYNC_SOURCE_DIR")
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
        self.graph_sync_source_dir = _default_graph_sync_source_dir()
        self.graph_checkpoint_dir = self.graph_sync_source_dir
        self.community_checkpoint_dir = None
        self.session_questions: dict[str, list[str]] = {}
        self.graph_backend: Any = None
        self.query_backend_name = "neo4j_live"
        self._neo4j_driver = None

        self._init_neo4j()

        self.answer_llm = _maybe_build_openai_llm("PIPELINE_ANSWER", "gpt-4o-mini", 400)

    def _init_neo4j(self) -> None:
        last_exc: Exception | None = None
        delay_seconds = 2.0

        for attempt in range(1, 4):
            driver = create_driver()
            try:
                if not test_connection(driver):
                    raise RuntimeError("Neo4j connection test failed.")
                stats = get_neo4j_graph_stats(driver)
                (
                    custom_entities,
                    entity_id_to_node,
                    custom_relations,
                    relation_metadata,
                    entity_descriptions,
                    entity_sources,
                    relation_index,
                    chunk_citation_map,
                    chunk_payload_map,
                ) = restore_from_neo4j(driver)
                if not custom_entities:
                    raise RuntimeError(
                        "Neo4j is connected but the graph is empty. "
                        "Sync the merged checkpoint first, for example: "
                        "python pipeline.py --graph-checkpoint-dir "
                        f"{self.graph_sync_source_dir} --community-checkpoint-dir "
                        f"{self.graph_sync_source_dir} --write-neo4j"
                    )

                self._neo4j_driver = driver
                self.graph_backend = Neo4jGraphQueryBackend(driver)
                self.query_backend_name = "neo4j_live"
                self.graph_meta = {
                    "source": "neo4j",
                    "database": os.getenv("NEO4J_DATABASE", "neo4j"),
                    "runtime_mode": "neo4j_only",
                    "graph_sync_source_dir": self.graph_sync_source_dir,
                    "live_queries": True,
                    **stats,
                }
                self.artifacts = GraphArtifacts(
                    custom_entities=custom_entities,
                    entity_id_to_node=entity_id_to_node,
                    custom_relations=custom_relations,
                    relation_metadata=relation_metadata,
                    entity_descriptions=entity_descriptions,
                    entity_sources=entity_sources,
                    relation_index=relation_index,
                    chunk_citation_map=chunk_citation_map,
                    chunk_payload_map=chunk_payload_map,
                )
                self.communities = load_or_build_communities(
                    self.artifacts,
                    checkpoint_dir=None,
                    force_rebuild=True,
                )
                self.summaries = load_or_build_community_summaries(
                    self.communities,
                    self.artifacts,
                    checkpoint_dir=None,
                    llm_client=None,
                    force_rebuild=True,
                )
                return
            except Exception as exc:
                driver.close()
                last_exc = exc
                if attempt >= 3:
                    raise
                print(
                    f"Neo4j init attempt {attempt}/3 failed: {exc}. "
                    f"Retrying in {delay_seconds:.1f}s..."
                )
                time.sleep(delay_seconds)
                delay_seconds *= 2.0

        if last_exc is not None:
            raise last_exc

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
            graph_backend=self.graph_backend,
        )
        if normalized_session_id:
            updated_history = [*history, question][-6:]
            self.session_questions[normalized_session_id] = updated_history
        return result

    def close(self) -> None:
        if self._neo4j_driver is not None:
            self._neo4j_driver.close()
            self._neo4j_driver = None
        self.graph_backend = None


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
        "graph_sync_source_dir": service.graph_sync_source_dir,
        "community_checkpoint_dir": service.community_checkpoint_dir,
        "query_backend": service.query_backend_name,
        "entity_count": service.artifacts.entity_count,
        "relation_count": service.artifacts.relation_count,
        "chunk_count": len(service.artifacts.chunk_payload_map),
        "community_count": len(service.communities),
        "summary_count": len(service.summaries),
    }


@app.get("/meta")
def meta() -> dict[str, Any]:
    service = get_service()
    return {
        "graph_meta": service.graph_meta,
        "graph_sync_source_dir": service.graph_sync_source_dir,
        "query_backend": service.query_backend_name,
        "entity_count": service.artifacts.entity_count,
        "relation_count": service.artifacts.relation_count,
        "chunk_count": len(service.artifacts.chunk_payload_map),
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


@app.on_event("shutdown")
def shutdown_event() -> None:
    if get_service.cache_info().currsize:
        service = get_service()
        service.close()
        get_service.cache_clear()
