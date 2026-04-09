from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from graphrag_pipeline import (
    clean_graph_checkpoint,
    repair_graph_checkpoint,
    extract_graph_from_nodes,
    graph_checkpoint_exists,
    hybrid_query,
    load_graph_checkpoint,
    load_or_build_communities,
    load_or_build_community_summaries,
    structural_checks,
    summarize_check_results,
)
from pipeline_helpers import (
    enrich_nodes_with_context,
    load_enriched_nodes_checkpoint,
    load_or_build_markdown_nodes,
)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _maybe_build_openai_llm(prefix: str, default_model: str, default_max_tokens: int) -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv(f"{prefix}_MODEL", default_model)
    max_tokens = _int_env(f"{prefix}_MAX_TOKENS", default_max_tokens)
    reasoning_effort = os.getenv(f"{prefix}_REASONING_EFFORT")

    from llama_index.llms.openai import OpenAI

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        return OpenAI(**kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        return OpenAI(**kwargs)


def _print_pipeline_stats(graph_meta: dict[str, Any], artifacts, communities, summaries) -> None:
    print("=" * 72)
    print("GRAPH RAG PIPELINE")
    print("=" * 72)
    print(f"Entities:            {artifacts.entity_count}")
    print(f"Relations:           {artifacts.relation_count}")
    print(f"Relation index keys: {len(artifacts.relation_index)}")
    print(f"Chunk citations:     {len(artifacts.chunk_citation_map)}")
    print(f"Communities:         {len(communities)}")
    print(f"Community summaries: {len(summaries)}")
    if graph_meta:
        print(f"Graph checkpoint:    {graph_meta}")
    print()


def _print_stage(title: str, detail: str | None = None) -> None:
    print("\n" + "=" * 72, flush=True)
    print(title, flush=True)
    if detail:
        print(detail, flush=True)
    print("=" * 72, flush=True)


def _run_smoke_tests(artifacts, summaries, answer_llm=None, answer_with_llm: bool = False) -> int:
    print("=" * 72)
    print("STRUCTURAL TESTS")
    print("=" * 72)
    results = structural_checks(artifacts, summaries)
    for name, ok, detail in results:
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] {name} :: {detail}")
    passed, failed = summarize_check_results(results)
    print(f"\nSummary: {passed} passed / {failed} failed\n")

    queries = [
        "I have fatigue and sadness, what conditions could this be?",
        "What is the difference between depression and bipolar disorder?",
        "What are the symptoms of PTSD?",
    ]
    for query in queries:
        print("=" * 72)
        print(f"QUERY: {query}")
        print("=" * 72)
        result = hybrid_query(
            query,
            artifacts,
            community_summaries=summaries,
            llm_client=answer_llm,
            answer_with_llm=answer_with_llm,
        )
        print(result["answer"])
        if result["citations"]:
            print("\nCitations:")
            for citation in result["citations"]:
                print(f"- {citation}")
        print()

    return failed


def _sync_to_neo4j(artifacts) -> None:
    from neo4j_helpers import (
        clear_neo4j_graph,
        create_driver,
        create_neo4j_indexes,
        test_connection,
        write_graph_to_neo4j,
    )

    driver = create_driver()
    try:
        if not test_connection(driver):
            raise RuntimeError("Neo4j connection test failed.")
        print("Clearing existing Neo4j graph before sync...")
        clear_neo4j_graph(driver)
        create_neo4j_indexes(driver)
        write_graph_to_neo4j(
            driver,
            artifacts.custom_entities,
            set(artifacts.relation_metadata.keys()),
            artifacts.relation_metadata,
            artifacts.entity_sources,
            artifacts.chunk_citation_map,
        )
        print("Neo4j sync complete.")
    finally:
        driver.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the notebook workflow as a reusable GraphRAG pipeline.")
    parser.add_argument("--input-dir", default="./data/markdown_filtered")
    parser.add_argument("--source-manifest-path", default=None)
    parser.add_argument("--node-checkpoint-dir", default="./checkpoints/pipeline")
    parser.add_argument("--graph-checkpoint-dir", default="./checkpoints/extraction_v4")
    parser.add_argument("--community-checkpoint-dir", default="./checkpoints/extraction_v4")
    parser.add_argument("--force-rebuild-nodes", action="store_true")
    parser.add_argument("--force-rebuild-enrichment", action="store_true")
    parser.add_argument("--force-rebuild-graph", action="store_true")
    parser.add_argument("--clean-graph-checkpoint-dir", default=None)
    parser.add_argument("--force-rebuild-clean-graph", action="store_true")
    parser.add_argument("--repair-graph-checkpoint-dir", default=None)
    parser.add_argument("--force-rebuild-repair-graph", action="store_true")
    parser.add_argument("--repair-dsm-gaps", action="store_true")
    parser.add_argument("--force-rebuild-communities", action="store_true")
    parser.add_argument("--no-resume-graph", action="store_true")
    parser.add_argument("--max-graph-nodes", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--write-neo4j", action="store_true")
    parser.add_argument("--query", default=None)
    parser.add_argument("--answer-with-llm", action="store_true")
    parser.add_argument("--summary-with-llm", action="store_true")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    load_dotenv(Path(".env"), override=False)

    graph_exists = graph_checkpoint_exists(args.graph_checkpoint_dir)
    need_source_pipeline = args.force_rebuild_graph or not graph_exists
    enriched_nodes_cache = None

    graph_meta: dict[str, Any]
    if need_source_pipeline:
        _print_stage("STAGE 1: MARKDOWN NODES", args.input_dir)
        documents, nodes = load_or_build_markdown_nodes(
            input_dir=args.input_dir,
            checkpoint_dir=args.node_checkpoint_dir,
            source_manifest_path=args.source_manifest_path,
            force_rebuild=args.force_rebuild_nodes,
        )
        print(f"Loaded {len(documents)} documents and {len(nodes)} markdown nodes.")

        enrich_llm = _maybe_build_openai_llm("PIPELINE_ENRICH", "gpt-4o-mini", 60)
        if enrich_llm is None:
            raise RuntimeError(
                "Graph extraction needs enriched nodes, but OPENAI_API_KEY is not configured."
            )

        _print_stage("STAGE 2: ENRICH NODES", f"checkpoint: {args.node_checkpoint_dir}")
        enriched_nodes = enrich_nodes_with_context(
            nodes,
            llm_client=enrich_llm,
            checkpoint_dir=args.node_checkpoint_dir,
            resume=True,
            force_rebuild=args.force_rebuild_enrichment,
            save_every=25,
            progress_every=args.progress_every,
        )
        print(f"Prepared {len(enriched_nodes)} enriched nodes.")
        enriched_nodes_cache = enriched_nodes

        graph_llm = _maybe_build_openai_llm("PIPELINE_GRAPH", "gpt-4o-mini", 900)
        if graph_llm is None:
            raise RuntimeError(
                "Graph extraction requires an LLM, but OPENAI_API_KEY is not configured."
            )

        _print_stage("STAGE 3: EXTRACT GRAPH", f"checkpoint: {args.graph_checkpoint_dir}")
        artifacts, graph_meta = extract_graph_from_nodes(
            enriched_nodes,
            graph_llm,
            checkpoint_dir=args.graph_checkpoint_dir,
            force_rebuild=args.force_rebuild_graph,
            resume=not args.no_resume_graph,
            progress_every=args.progress_every,
            max_nodes=args.max_graph_nodes,
        )
    else:
        _print_stage("STAGE 1: LOAD GRAPH CHECKPOINT", args.graph_checkpoint_dir)
        artifacts, graph_meta = load_graph_checkpoint(args.graph_checkpoint_dir)
        print(f"Loaded graph checkpoint from {args.graph_checkpoint_dir}.")

    active_graph_dir = args.graph_checkpoint_dir
    if args.clean_graph_checkpoint_dir:
        active_graph_dir = args.clean_graph_checkpoint_dir
        clean_exists = graph_checkpoint_exists(active_graph_dir)
        need_clean_graph = args.force_rebuild_clean_graph or not clean_exists
        _print_stage(
            "STAGE 3B: CLEAN GRAPH",
            f"source: {args.graph_checkpoint_dir}\ntarget: {active_graph_dir}",
        )
        if need_clean_graph:
            if enriched_nodes_cache is None:
                enriched_nodes_cache = load_enriched_nodes_checkpoint(args.node_checkpoint_dir)
            artifacts, graph_meta, cleanup_report = clean_graph_checkpoint(
                args.graph_checkpoint_dir,
                active_graph_dir,
                enriched_nodes=enriched_nodes_cache,
            )
            print(f"Cleanup report: {cleanup_report}")
        else:
            artifacts, graph_meta = load_graph_checkpoint(active_graph_dir)
            cleanup_report = graph_meta.get("cleanup_report", {})
            print(f"Loaded cleaned graph checkpoint from {active_graph_dir}.")
            if cleanup_report:
                print(f"Cleanup report: {cleanup_report}")

    if args.repair_dsm_gaps:
        if not args.repair_graph_checkpoint_dir:
            raise RuntimeError("--repair-dsm-gaps requires --repair-graph-checkpoint-dir.")
        repair_exists = graph_checkpoint_exists(args.repair_graph_checkpoint_dir)
        active_graph_dir = args.repair_graph_checkpoint_dir
        _print_stage(
            "STAGE 3C: REPAIR GRAPH",
            f"source: {args.clean_graph_checkpoint_dir or args.graph_checkpoint_dir}\n"
            f"target: {active_graph_dir}",
        )
        if args.force_rebuild_repair_graph or not repair_exists:
            repair_llm = _maybe_build_openai_llm("PIPELINE_REPAIR", "gpt-4o-mini", 900)
            if repair_llm is None:
                raise RuntimeError(
                    "Graph repair requires an LLM, but OPENAI_API_KEY is not configured."
                )
            if enriched_nodes_cache is None:
                enriched_nodes_cache = load_enriched_nodes_checkpoint(args.node_checkpoint_dir)
            artifacts, graph_meta, repair_report = repair_graph_checkpoint(
                args.clean_graph_checkpoint_dir or args.graph_checkpoint_dir,
                active_graph_dir,
                enriched_nodes_cache,
                repair_llm,
                progress_every=max(1, min(args.progress_every, 5)),
            )
            print(f"Repair report: {repair_report}")
        else:
            artifacts, graph_meta = load_graph_checkpoint(active_graph_dir)
            repair_report = graph_meta.get("repair_report", {})
            print(f"Loaded repaired graph checkpoint from {active_graph_dir}.")
            if repair_report:
                print(f"Repair report: {repair_report}")

    summary_llm = None
    if args.summary_with_llm:
        summary_llm = _maybe_build_openai_llm("PIPELINE_SUMMARY", "gpt-4o-mini", 220)

    community_checkpoint_dir = args.community_checkpoint_dir
    if args.clean_graph_checkpoint_dir and community_checkpoint_dir == args.graph_checkpoint_dir:
        community_checkpoint_dir = active_graph_dir
    if args.repair_dsm_gaps and community_checkpoint_dir in {
        args.graph_checkpoint_dir,
        args.clean_graph_checkpoint_dir,
    }:
        community_checkpoint_dir = active_graph_dir

    _print_stage("STAGE 4: COMMUNITIES", f"checkpoint: {community_checkpoint_dir}")
    communities = load_or_build_communities(
        artifacts,
        checkpoint_dir=community_checkpoint_dir,
        force_rebuild=args.force_rebuild_communities,
    )
    summaries = load_or_build_community_summaries(
        communities,
        artifacts,
        checkpoint_dir=community_checkpoint_dir,
        llm_client=summary_llm,
        force_rebuild=args.force_rebuild_communities,
        progress_every=max(1, min(args.progress_every, 5)),
    )

    _print_pipeline_stats(graph_meta, artifacts, communities, summaries)

    if args.write_neo4j:
        _print_stage("STAGE 5: SYNC NEO4J")
        _sync_to_neo4j(artifacts)

    answer_llm = None
    if args.answer_with_llm:
        answer_llm = _maybe_build_openai_llm("PIPELINE_ANSWER", "gpt-4o-mini", 400)

    if args.query:
        _print_stage("STAGE 6: QUERY", args.query)
        result = hybrid_query(
            args.query,
            artifacts,
            community_summaries=summaries,
            llm_client=answer_llm,
            answer_with_llm=args.answer_with_llm,
        )
        print(result["answer"])
        if result["citations"]:
            print("\nCitations:")
            for citation in result["citations"]:
                print(f"- {citation}")
        return 0

    if args.smoke_test:
        _print_stage("STAGE 6: SMOKE TEST")
        return _run_smoke_tests(
            artifacts,
            summaries,
            answer_llm=answer_llm,
            answer_with_llm=args.answer_with_llm,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
