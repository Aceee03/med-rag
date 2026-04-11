from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from llama_index.core.llms import ChatMessage

from graphrag_pipeline import _extract_response_text
from pipeline_helpers import citation_from_metadata


TOKEN_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "by",
    "that",
    "this",
    "it",
    "as",
    "at",
    "from",
    "what",
    "which",
    "who",
    "how",
}


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in TOKEN_STOPWORDS
    ]


def _excerpt(text: str, limit: int = 420) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _prepare_sentence_transformers_environment() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


@dataclass
class ChunkRecord:
    node_id: str
    citation: str
    text: str
    context_tag: str
    section_title: str
    source_label: str

    @property
    def combined_text(self) -> str:
        if self.context_tag:
            return f"[CONTEXT: {self.context_tag}] {self.text}"
        return self.text


@dataclass
class RetrievedChunk:
    node_id: str
    citation: str
    text: str
    score: float


def build_chunk_records(enriched_nodes: list[Any]) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for node in enriched_nodes:
        metadata = getattr(node, "metadata", {}) or {}
        citation = citation_from_metadata(
            metadata,
            fallback=str(metadata.get("citation") or "Unknown source"),
        )
        records.append(
            ChunkRecord(
                node_id=str(getattr(node, "node_id", "")),
                citation=str(citation),
                text=str(metadata.get("original_text") or getattr(node, "text", "") or ""),
                context_tag=str(metadata.get("context_tag", "") or ""),
                section_title=str(metadata.get("section_title", "") or ""),
                source_label=str(metadata.get("source_label", "") or ""),
            )
        )
    return records


class VectorRetriever:
    def __init__(
        self,
        records: list[ChunkRecord],
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.records = records
        self.model_name = model_name
        self.backend = "lexical"
        self.backend_error = ""
        self._embedder = None
        self._embeddings = None
        self._doc_tokens = [_tokenize(record.combined_text) for record in records]
        self._idf = self._build_idf(self._doc_tokens)

        try:
            _prepare_sentence_transformers_environment()
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(model_name)
            self._embeddings = np.asarray(
                self._embedder.encode(
                    [record.combined_text for record in records],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            self.backend = "sentence-transformers"
        except Exception as exc:
            self._embedder = None
            self._embeddings = None
            self.backend_error = str(exc)
            warnings.warn(
                f"VectorRetriever falling back to lexical search because '{model_name}' could not be loaded: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _build_idf(doc_tokens: list[list[str]]) -> dict[str, float]:
        df: dict[str, int] = {}
        doc_count = max(1, len(doc_tokens))
        for tokens in doc_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        return {
            token: math.log((doc_count + 1) / (freq + 1)) + 1.0
            for token, freq in df.items()
        }

    def _lexical_search(self, question: str, top_k: int) -> list[RetrievedChunk]:
        query_tokens = _tokenize(question)
        if not query_tokens:
            return []

        scored: list[RetrievedChunk] = []
        for record, tokens in zip(self.records, self._doc_tokens):
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                if token in tf:
                    score += (1.0 + math.log(1.0 + tf[token])) * self._idf.get(token, 1.0)
            if score <= 0:
                continue
            scored.append(
                RetrievedChunk(
                    node_id=record.node_id,
                    citation=record.citation,
                    text=record.combined_text,
                    score=score,
                )
            )
        scored.sort(key=lambda item: (-item.score, item.citation))
        return scored[:top_k]

    def search(self, question: str, top_k: int = 8) -> list[RetrievedChunk]:
        if self._embedder is not None and self._embeddings is not None:
            query_embedding = np.asarray(
                self._embedder.encode(
                    [question],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )[0]
            scores = self._embeddings @ query_embedding
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [
                RetrievedChunk(
                    node_id=self.records[index].node_id,
                    citation=self.records[index].citation,
                    text=self.records[index].combined_text,
                    score=float(scores[index]),
                )
                for index in top_indices
            ]
        return self._lexical_search(question, top_k)


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self.backend = "lexical"
        self.backend_error = ""
        self._model = None
        try:
            _prepare_sentence_transformers_environment()
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
            self.backend = "cross-encoder"
        except Exception as exc:
            self._model = None
            self.backend_error = str(exc)
            warnings.warn(
                f"CrossEncoderReranker falling back to lexical reranking because '{model_name}' could not be loaded: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def rerank(
        self,
        question: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if self._model is not None:
            pairs = [(question, chunk.text) for chunk in candidates]
            scores = self._model.predict(pairs)
            rescored = [
                RetrievedChunk(
                    node_id=chunk.node_id,
                    citation=chunk.citation,
                    text=chunk.text,
                    score=float(score),
                )
                for chunk, score in zip(candidates, scores)
            ]
            rescored.sort(key=lambda item: (-item.score, item.citation))
            return rescored[:top_k]

        query_tokens = set(_tokenize(question))
        rescored: list[RetrievedChunk] = []
        for chunk in candidates:
            text_tokens = set(_tokenize(chunk.text))
            overlap = len(query_tokens & text_tokens)
            rescored.append(
                RetrievedChunk(
                    node_id=chunk.node_id,
                    citation=chunk.citation,
                    text=chunk.text,
                    score=float(overlap) + chunk.score,
                )
            )
        rescored.sort(key=lambda item: (-item.score, item.citation))
        return rescored[:top_k]


def render_contexts(chunks: list[RetrievedChunk], *, max_chars: int = 700) -> list[str]:
    return [f"[{chunk.citation}]\n{_excerpt(chunk.text, max_chars)}" for chunk in chunks]


def answer_with_context(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    llm_client: Any = None,
) -> str:
    contexts = render_contexts(chunks)
    if llm_client is None:
        if not contexts:
            return "No relevant context was found."
        lines = [f"Question: {question}", "Retrieved context:"]
        for context in contexts[:4]:
            lines.append(f"- {context}")
        return "\n".join(lines)

    prompt = (
        "Answer the user's question using only the provided context. "
        "Be concise, clinically factual, and say when the context is insufficient.\n\n"
        f"QUESTION:\n{question}\n\n"
        "CONTEXT:\n"
        + "\n\n".join(contexts[:5])
        + "\n\nANSWER:"
    )
    response = llm_client.chat(
        [
            ChatMessage(
                role="system",
                content="You are a careful clinical-information assistant. Use only the supplied context.",
            ),
            ChatMessage(role="user", content=prompt),
        ]
    )
    return _extract_response_text(response).strip()
