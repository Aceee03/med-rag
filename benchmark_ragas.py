from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings(
    "ignore",
    message="The 'validate_default' attribute with value True was provided to the `Field\\(\\)` function.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"instructor\..*",
)

from dotenv import load_dotenv

from graphrag_pipeline import (
    hybrid_query,
    load_graph_checkpoint,
    load_or_build_communities,
    load_or_build_community_summaries,
)
from pipeline import _maybe_build_openai_llm
from pipeline_helpers import _build_nodes_hash, load_enriched_nodes_checkpoint
from retrieval_stack import (
    CrossEncoderReranker,
    VectorRetriever,
    answer_with_context,
    build_chunk_records,
    render_contexts,
)


def load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    cases: list[dict[str, Any]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    if not cases:
        raise ValueError(f"No evaluation cases found in {file_path}")
    return cases


def _eval_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _f1_score(predicted: set[str], gold: set[str]) -> float:
    overlap = len(predicted & gold)
    precision = _safe_ratio(overlap, len(predicted))
    recall = _safe_ratio(overlap, len(gold))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def graph_contexts(result: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    if result.get("multipath_result"):
        contexts.append(result["multipath_result"])
    if result.get("local_result"):
        contexts.append(result["local_result"])
    if result.get("global_result"):
        contexts.append(result["global_result"])
    if result.get("citations"):
        contexts.append("Citations:\n" + "\n".join(f"- {item}" for item in result["citations"]))
    return contexts


def run_systems(
    cases: list[dict[str, Any]],
    *,
    graph_artifacts,
    community_summaries,
    vector_retriever: VectorRetriever,
    reranker: CrossEncoderReranker,
    answer_llm: Any = None,
    vector_top_k: int = 8,
    rerank_top_k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    outputs = {
        "vector_baseline": [],
        "vector_reranked": [],
        "graph_rag": [],
    }

    for case in cases:
        question = case["question"]
        ground_truth = case["ground_truth"]

        vector_chunks = vector_retriever.search(question, top_k=vector_top_k)
        reranked_chunks = reranker.rerank(question, vector_chunks, top_k=rerank_top_k)

        vector_answer = answer_with_context(question, vector_chunks[:rerank_top_k], llm_client=answer_llm)
        reranked_answer = answer_with_context(question, reranked_chunks, llm_client=answer_llm)
        graph_result = hybrid_query(
            question,
            graph_artifacts,
            community_summaries=community_summaries,
            llm_client=answer_llm,
            answer_with_llm=answer_llm is not None,
        )

        outputs["vector_baseline"].append(
            {
                "question": question,
                "answer": vector_answer,
                "contexts": render_contexts(vector_chunks[:rerank_top_k]),
                "ground_truth": ground_truth,
            }
        )
        outputs["vector_reranked"].append(
            {
                "question": question,
                "answer": reranked_answer,
                "contexts": render_contexts(reranked_chunks),
                "ground_truth": ground_truth,
            }
        )
        outputs["graph_rag"].append(
            {
                "question": question,
                "answer": graph_result["answer"],
                "contexts": graph_contexts(graph_result),
                "ground_truth": ground_truth,
            }
        )

    return outputs


def save_predictions(output_dir: Path, results: dict[str, list[dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for system_name, rows in results.items():
        path = output_dir / f"{system_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _proxy_metric_summary(system_name: str, rows: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    answer_scores: list[float] = []
    context_precision_scores: list[float] = []
    context_recall_scores: list[float] = []
    faithfulness_scores: list[float] = []
    for row in rows:
        gold_tokens = _eval_tokens(row["ground_truth"])
        answer_tokens = _eval_tokens(row["answer"])
        context_tokens = _eval_tokens(" ".join(row.get("contexts", [])))
        answer_scores.append(_f1_score(answer_tokens, gold_tokens))
        context_precision_scores.append(_safe_ratio(len(context_tokens & gold_tokens), len(context_tokens)))
        context_recall_scores.append(_safe_ratio(len(context_tokens & gold_tokens), len(gold_tokens)))
        faithfulness_scores.append(_safe_ratio(len(answer_tokens & context_tokens), len(answer_tokens)))
    return {
        "system": system_name,
        "status": "proxy",
        "reason": reason,
        "metric_backend": "proxy-token-overlap",
        "answer_relevancy": sum(answer_scores) / max(1, len(answer_scores)),
        "context_precision": sum(context_precision_scores) / max(1, len(context_precision_scores)),
        "context_recall": sum(context_recall_scores) / max(1, len(context_recall_scores)),
        "faithfulness": sum(faithfulness_scores) / max(1, len(faithfulness_scores)),
    }


def evaluate_with_ragas(
    system_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from openai import OpenAI
    except Exception as exc:
        return _proxy_metric_summary(system_name, rows, f"ragas import failed: {exc}")

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return _proxy_metric_summary(system_name, rows, "OPENAI_API_KEY not configured for ragas metrics")
        client = OpenAI(api_key=api_key)
        judge_llm = llm_factory("gpt-4o-mini", client=client)
        judge_embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=row["question"],
                    retrieved_contexts=list(row.get("contexts", [])),
                    response=row["answer"],
                    reference=row["ground_truth"],
                )
                for row in rows
            ]
        )
        result = evaluate(
            dataset=dataset,
            metrics=[
                type(answer_relevancy)(llm=judge_llm, embeddings=judge_embeddings),
                type(context_precision)(llm=judge_llm),
                type(context_recall)(llm=judge_llm),
                type(faithfulness)(llm=judge_llm),
            ],
            raise_exceptions=False,
            show_progress=False,
        )
    except Exception as exc:
        return _proxy_metric_summary(system_name, rows, f"ragas evaluate failed: {exc}")

    summary: dict[str, Any]
    if hasattr(result, "to_dict"):
        summary = dict(result.to_dict())
    elif isinstance(result, dict):
        summary = dict(result)
    else:
        summary = {"raw_result": str(result)}
    metric_values = [
        summary.get("answer_relevancy"),
        summary.get("context_precision"),
        summary.get("context_recall"),
        summary.get("faithfulness"),
    ]
    if not any(isinstance(value, (int, float)) for value in metric_values):
        return _proxy_metric_summary(system_name, rows, "ragas returned no numeric metric outputs")
    summary["system"] = system_name
    summary["status"] = "ok"
    summary["metric_backend"] = "ragas"
    return summary


def write_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ragas_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metric_columns = ["answer_relevancy", "context_precision", "context_recall", "faithfulness"]
    lines = [
        "# Thesis Benchmark Summary",
        "",
        "| System | Status | Answer Relevancy | Context Precision | Context Recall | Faithfulness |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        values = [
            summary.get("system", ""),
            summary.get("status", ""),
        ]
        for column in metric_columns:
            value = summary.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
        if summary.get("reason"):
            lines.append("")
            prefix = "Proxy metrics for" if summary.get("status") == "proxy" else "Skipped"
            lines.append(f"{prefix} `{summary.get('system')}`: {summary['reason']}")
    (output_dir / "ragas_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Vector RAG vs GraphRAG using RAGAS.")
    parser.add_argument("--node-checkpoint-dir", default="./checkpoints/pipeline")
    parser.add_argument("--graph-checkpoint-dir", default="./checkpoints/dsm_graph_v6_repaired")
    parser.add_argument("--community-checkpoint-dir", default="./checkpoints/dsm_graph_v6_repaired")
    parser.add_argument("--eval-cases", default="./data/eval/thesis_eval_cases.jsonl")
    parser.add_argument("--output-dir", default="./checkpoints/benchmark")
    parser.add_argument("--vector-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--vector-top-k", type=int, default=8)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--answer-with-llm", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--proxy-only", action="store_true")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    load_dotenv(Path(".env"), override=False)

    if hasattr(os.sys.stdout, "reconfigure"):
        os.sys.stdout.reconfigure(line_buffering=True)

    enriched_nodes = load_enriched_nodes_checkpoint(args.node_checkpoint_dir)
    enriched_nodes_hash = _build_nodes_hash(enriched_nodes)
    chunk_records = build_chunk_records(enriched_nodes)
    vector_retriever = VectorRetriever(chunk_records, model_name=args.vector_model)
    reranker = CrossEncoderReranker(model_name=args.cross_encoder_model)

    graph_artifacts, graph_meta = load_graph_checkpoint(args.graph_checkpoint_dir)
    graph_nodes_hash = graph_meta.get("nodes_hash")
    if graph_nodes_hash and graph_nodes_hash != enriched_nodes_hash:
        raise RuntimeError(
            "Benchmark node checkpoint does not match the graph checkpoint. "
            f"Expected nodes_hash={graph_nodes_hash}, got {enriched_nodes_hash}. "
            "Pass the matching --node-checkpoint-dir for this graph."
        )
    communities = load_or_build_communities(
        graph_artifacts,
        checkpoint_dir=args.community_checkpoint_dir,
        force_rebuild=False,
    )
    summaries = load_or_build_community_summaries(
        communities,
        graph_artifacts,
        checkpoint_dir=args.community_checkpoint_dir,
        llm_client=None,
        force_rebuild=False,
    )

    answer_llm = None
    if args.answer_with_llm:
        answer_llm = _maybe_build_openai_llm("PIPELINE_ANSWER", "gpt-4o-mini", 400)

    cases = load_eval_cases(args.eval_cases)
    results = run_systems(
        cases,
        graph_artifacts=graph_artifacts,
        community_summaries=summaries,
        vector_retriever=vector_retriever,
        reranker=reranker,
        answer_llm=answer_llm,
        vector_top_k=args.vector_top_k,
        rerank_top_k=args.rerank_top_k,
    )

    output_dir = Path(args.output_dir)
    save_predictions(output_dir, results)

    print(f"Vector retriever backend: {vector_retriever.backend}")
    print(f"Reranker backend: {reranker.backend}")
    for system_name, rows in results.items():
        print(f"Saved {len(rows)} predictions for {system_name} -> {output_dir / f'{system_name}.jsonl'}")

    summaries_out: list[dict[str, Any]] = []
    if args.proxy_only:
        summaries_out = [
            _proxy_metric_summary(name, rows, "proxy-only flag set")
            for name, rows in results.items()
        ]
    elif args.skip_ragas:
        summaries_out = [
            {"system": name, "status": "skipped", "reason": "skip-ragas flag set"}
            for name in results
        ]
    else:
        for system_name, rows in results.items():
            print(f"Running RAGAS for {system_name}...")
            summaries_out.append(evaluate_with_ragas(system_name, rows))

    write_summary(output_dir, summaries_out)
    print(f"Wrote benchmark summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
