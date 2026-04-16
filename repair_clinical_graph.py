from __future__ import annotations

import argparse
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="The 'validate_default' attribute with value True was provided to the `Field\\(\\)` function.*",
)

from dotenv import load_dotenv
from llama_index.core.graph_stores.types import Relation

from graphrag_pipeline import (
    CHECKPOINT_VERSION,
    CLINICAL_REPAIR_TARGETS,
    _apply_extracted_payload_to_state,
    _graph_artifacts_from_state,
    _graph_state_from_artifacts,
    build_relation_index,
    clean_graph_artifacts,
    load_graph_checkpoint,
    refresh_chunk_citation_map,
    repair_graph_artifacts,
    save_graph_checkpoint,
    select_repair_nodes,
)
from llm_utils import _maybe_build_openai_llm
from pipeline_helpers import load_enriched_nodes_checkpoint


CURATED_RELATION_RESETS = [
    ("DEPRESSION", "HAS_PREVALENCE"),
    ("GENERALIZED ANXIETY DISORDER", "HAS_PREVALENCE"),
    ("GENERALIZED ANXIETY DISORDER", "DIFFERENTIAL_DIAGNOSIS"),
    ("POST TRAUMATIC STRESS DISORDER", "DIFFERENTIAL_DIAGNOSIS"),
    ("AUTISM SPECTRUM DISORDER", "TREATED_BY"),
]

CURATED_PAYLOADS = [
    {
        "condition": "DEPRESSION",
        "relations": ["HAS_PREVALENCE"],
        "source_hints": ["who__depression__who-depression-html", "An estimated 4% of the population experience depression"],
        "payload": {
            "entities": [
                {"id": "c0", "name": "DEPRESSION", "type": "CONDITION", "description": ""},
                {
                    "id": "p1",
                    "name": "GLOBAL PREVALENCE OF DEPRESSION",
                    "type": "PREVALENCE_STATEMENT",
                    "description": "An estimated 4% of the global population experience depression, about 332 million people worldwide.",
                },
                {
                    "id": "p2",
                    "name": "DEPRESSION MORE COMMON AMONG WOMEN",
                    "type": "PREVALENCE_STATEMENT",
                    "description": "Depression is about 1.5 times more common among women than among men.",
                },
            ],
            "relations": [
                {
                    "source": "c0",
                    "target": "p1",
                    "relation": "HAS_PREVALENCE",
                    "description": "An estimated 4% of the global population experience depression, about 332 million people worldwide.",
                    "strength": 10,
                },
                {
                    "source": "c0",
                    "target": "p2",
                    "relation": "HAS_PREVALENCE",
                    "description": "Depression is about 1.5 times more common among women than among men.",
                    "strength": 8,
                },
            ],
        },
    },
    {
        "condition": "GENERALIZED ANXIETY DISORDER",
        "relations": ["HAS_PREVALENCE", "DIFFERENTIAL_DIAGNOSIS"],
        "source_hints": [
            "The 12-month prevalence of generalized anxiety disorder is 0.9% among adolescents and 2.9% among adults",
            "Individuals with social anxiety disorder often have anticipatory anxiety",
            "Panic attacks that are triggered by worry in generalized anxiety disorder",
        ],
        "payload": {
            "entities": [
                {"id": "c0", "name": "GENERALIZED ANXIETY DISORDER", "type": "CONDITION", "description": ""},
                {
                    "id": "p1",
                    "name": "TWELVE MONTH PREVALENCE OF GENERALIZED ANXIETY DISORDER",
                    "type": "PREVALENCE_STATEMENT",
                    "description": "The 12-month prevalence of generalized anxiety disorder is 2.9% among adults and 0.9% among adolescents in the United States, with a worldwide mean of 1.3%.",
                },
                {
                    "id": "d1",
                    "name": "PANIC DISORDER",
                    "type": "CONDITION",
                    "description": "Unexpected panic attacks and persistent concern about the attacks support panic disorder rather than generalized anxiety disorder.",
                },
                {
                    "id": "d2",
                    "name": "SOCIAL ANXIETY DISORDER",
                    "type": "CONDITION",
                    "description": "Anxiety focused on social situations or evaluation by others suggests social anxiety disorder rather than generalized anxiety disorder.",
                },
                {
                    "id": "d3",
                    "name": "SEPARATION ANXIETY DISORDER",
                    "type": "CONDITION",
                    "description": "Worry limited to separation from attachment figures suggests separation anxiety disorder rather than generalized anxiety disorder.",
                },
                {
                    "id": "d4",
                    "name": "OBSESSIVE COMPULSIVE DISORDER",
                    "type": "CONDITION",
                    "description": "Intrusive unwanted obsessions suggest obsessive-compulsive disorder rather than generalized anxiety disorder.",
                },
                {
                    "id": "d5",
                    "name": "POST TRAUMATIC STRESS DISORDER",
                    "type": "CONDITION",
                    "description": "Anxiety and worry better explained by trauma-related symptoms suggest post-traumatic stress disorder rather than generalized anxiety disorder.",
                },
            ],
            "relations": [
                {
                    "source": "c0",
                    "target": "p1",
                    "relation": "HAS_PREVALENCE",
                    "description": "The 12-month prevalence of generalized anxiety disorder is 2.9% among adults and 0.9% among adolescents in the United States, with a worldwide mean of 1.3%.",
                    "strength": 10,
                },
                {
                    "source": "c0",
                    "target": "d1",
                    "relation": "DIFFERENTIAL_DIAGNOSIS",
                    "description": "Unexpected panic attacks and persistent concern about the attacks support panic disorder rather than generalized anxiety disorder.",
                    "strength": 9,
                },
                {
                    "source": "c0",
                    "target": "d2",
                    "relation": "DIFFERENTIAL_DIAGNOSIS",
                    "description": "Anxiety focused on social situations or evaluation by others suggests social anxiety disorder rather than generalized anxiety disorder.",
                    "strength": 9,
                },
                {
                    "source": "c0",
                    "target": "d3",
                    "relation": "DIFFERENTIAL_DIAGNOSIS",
                    "description": "Worry limited to separation from attachment figures suggests separation anxiety disorder rather than generalized anxiety disorder.",
                    "strength": 9,
                },
                {
                    "source": "c0",
                    "target": "d4",
                    "relation": "DIFFERENTIAL_DIAGNOSIS",
                    "description": "Intrusive unwanted obsessions suggest obsessive-compulsive disorder rather than generalized anxiety disorder.",
                    "strength": 9,
                },
                {
                    "source": "c0",
                    "target": "d5",
                    "relation": "DIFFERENTIAL_DIAGNOSIS",
                    "description": "Anxiety and worry better explained by trauma-related symptoms suggest post-traumatic stress disorder rather than generalized anxiety disorder.",
                    "strength": 8,
                },
            ],
        },
    },
    {
        "condition": "POST TRAUMATIC STRESS DISORDER",
        "relations": ["DIFFERENTIAL_DIAGNOSIS"],
        "source_hints": [
            "Acute stress disorder is distinguished from PTSD",
            "Major depressive disorder does not include any PTSD Criterion B or C symptoms",
            "Both ADHD and PTSD may include problems in attention",
        ],
        "payload": {
            "entities": [
                {"id": "c0", "name": "POST TRAUMATIC STRESS DISORDER", "type": "CONDITION", "description": ""},
                {
                    "id": "d1",
                    "name": "ACUTE STRESS DISORDER",
                    "type": "CONDITION",
                    "description": "Acute stress disorder is more appropriate when the symptom pattern is limited to 3 days to 1 month after trauma.",
                },
                {
                    "id": "d2",
                    "name": "ADJUSTMENT DISORDER",
                    "type": "CONDITION",
                    "description": "Adjustment disorder is more appropriate when the response to a stressor does not meet PTSD criteria or the stressor does not satisfy Criterion A.",
                },
                {
                    "id": "d3",
                    "name": "OBSESSIVE COMPULSIVE DISORDER",
                    "type": "CONDITION",
                    "description": "Intrusive thoughts that are obsessions rather than trauma memories suggest obsessive-compulsive disorder rather than PTSD.",
                },
                {
                    "id": "d4",
                    "name": "MAJOR DEPRESSIVE DISORDER",
                    "type": "CONDITION",
                    "description": "Major depressive disorder does not include the PTSD intrusion and avoidance symptoms.",
                },
                {
                    "id": "d5",
                    "name": "ATTENTION DEFICIT HYPERACTIVITY DISORDER",
                    "type": "CONDITION",
                    "description": "Attention and concentration problems that predate trauma exposure suggest ADHD rather than PTSD.",
                },
            ],
            "relations": [
                {
                    "source": "c0",
                    "target": "d1",
                    "relation": "DIFFERENTIAL_DIAGNOSIS",
                    "description": "Acute stress disorder is more appropriate when the symptom pattern is limited to 3 days to 1 month after trauma.",
                    "strength": 10,
                },
                {
                    "source": "c0",
                    "target": "d2",
                    "relation": "DIFFERENTIAL_DIAGNOSIS",
                    "description": "Adjustment disorder is more appropriate when the response to a stressor does not meet PTSD criteria or the stressor does not satisfy Criterion A.",
                    "strength": 9,
                },
                {
                    "source": "c0",
                    "target": "d3",
                    "relation": "DIFFERENTIAL_DIAGNOSIS",
                    "description": "Intrusive thoughts that are obsessions rather than trauma memories suggest obsessive-compulsive disorder rather than PTSD.",
                    "strength": 9,
                },
                {
                    "source": "c0",
                    "target": "d4",
                    "relation": "DIFFERENTIAL_DIAGNOSIS",
                    "description": "Major depressive disorder does not include the PTSD intrusion and avoidance symptoms.",
                    "strength": 8,
                },
                {
                    "source": "c0",
                    "target": "d5",
                    "relation": "DIFFERENTIAL_DIAGNOSIS",
                    "description": "Attention and concentration problems that predate trauma exposure suggest ADHD rather than PTSD.",
                    "strength": 8,
                },
            ],
        },
    },
    {
        "condition": "AUTISM SPECTRUM DISORDER",
        "relations": ["TREATED_BY"],
        "source_hints": [
            "behavioral, psychological, educational, occupational, physical, or speech-language therapy",
            "Evidence-based psychosocial interventions can improve communication and social skills",
        ],
        "payload": {
            "entities": [
                {"id": "c0", "name": "AUTISM SPECTRUM DISORDER", "type": "CONDITION", "description": ""},
                {
                    "id": "t1",
                    "name": "BEHAVIORAL THERAPY",
                    "type": "TREATMENT",
                    "description": "Structured behavioral interventions can help autistic people build adaptive skills and reduce challenging behaviors.",
                },
                {
                    "id": "t2",
                    "name": "PSYCHOLOGICAL THERAPY",
                    "type": "TREATMENT",
                    "description": "Psychological interventions can support emotional regulation, coping, and daily functioning in autistic people.",
                },
                {
                    "id": "t3",
                    "name": "EDUCATIONAL SUPPORT",
                    "type": "TREATMENT",
                    "description": "Educational services and structured learning supports can help address communication, learning, and daily-living needs.",
                },
                {
                    "id": "t4",
                    "name": "OCCUPATIONAL THERAPY",
                    "type": "TREATMENT",
                    "description": "Occupational therapy can support sensory, daily-living, and functional skills.",
                },
                {
                    "id": "t5",
                    "name": "SPEECH LANGUAGE THERAPY",
                    "type": "TREATMENT",
                    "description": "Speech-language therapy can improve communication skills and social interaction.",
                },
                {
                    "id": "t6",
                    "name": "PSYCHOSOCIAL INTERVENTIONS",
                    "type": "TREATMENT",
                    "description": "Evidence-based psychosocial interventions can improve communication and social skills and support quality of life.",
                },
            ],
            "relations": [
                {
                    "source": "c0",
                    "target": "t1",
                    "relation": "TREATED_BY",
                    "description": "Structured behavioral interventions can help autistic people build adaptive skills and reduce challenging behaviors.",
                    "strength": 9,
                },
                {
                    "source": "c0",
                    "target": "t2",
                    "relation": "TREATED_BY",
                    "description": "Psychological interventions can support emotional regulation, coping, and daily functioning in autistic people.",
                    "strength": 8,
                },
                {
                    "source": "c0",
                    "target": "t3",
                    "relation": "TREATED_BY",
                    "description": "Educational services and structured learning supports can help address communication, learning, and daily-living needs.",
                    "strength": 8,
                },
                {
                    "source": "c0",
                    "target": "t4",
                    "relation": "TREATED_BY",
                    "description": "Occupational therapy can support sensory, daily-living, and functional skills.",
                    "strength": 8,
                },
                {
                    "source": "c0",
                    "target": "t5",
                    "relation": "TREATED_BY",
                    "description": "Speech-language therapy can improve communication skills and social interaction.",
                    "strength": 8,
                },
                {
                    "source": "c0",
                    "target": "t6",
                    "relation": "TREATED_BY",
                    "description": "Evidence-based psychosocial interventions can improve communication and social skills and support quality of life.",
                    "strength": 9,
                },
            ],
        },
    },
]


def _dedupe_nodes(*node_groups: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen_ids: set[str] = set()
    for group in node_groups:
        for node in group:
            node_id = str(getattr(node, "node_id", "") or "")
            if not node_id or node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            deduped.append(node)
    return deduped


def _remove_relation_family(state: dict[str, Any], subject: str, relation_label: str) -> None:
    retained_metadata = {
        key: value
        for key, value in state["relation_metadata"].items()
        if not (key[0] == subject and key[1] == relation_label)
    }
    state["relation_metadata"] = retained_metadata
    state["seen_relations"] = set(retained_metadata.keys())


def _rebuild_state_graph(state: dict[str, Any]) -> None:
    custom_relations: list[Relation] = []
    for subject, relation_label, target in state["relation_metadata"]:
        subject_entity = state["custom_entities"].get(subject)
        target_entity = state["custom_entities"].get(target)
        if subject_entity is None or target_entity is None:
            continue
        custom_relations.append(
            Relation(
                source_id=subject_entity.id,
                target_id=target_entity.id,
                label=relation_label,
            )
        )
    state["custom_relations"] = custom_relations
    state["relation_index"] = build_relation_index(
        state["custom_entities"],
        custom_relations,
        state["entity_id_to_node"],
    )


def _select_curated_chunk_ids(
    nodes: list[Any],
    artifacts,
    condition_name: str,
    relation_labels: list[str],
    source_hints: list[str],
    *,
    top_k: int,
) -> set[str]:
    selected: set[str] = set()
    lowered_hints = [hint.lower() for hint in source_hints]
    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        haystack = " ".join(
            [
                str(metadata.get("file_name", "")),
                str(metadata.get("source_label", "")),
                str(metadata.get("citation", "")),
                str(metadata.get("section_title", "")),
                str(metadata.get("original_text", ""))[:6000],
            ]
        ).lower()
        if any(hint in haystack for hint in lowered_hints):
            node_id = str(getattr(node, "node_id", "") or "")
            if node_id:
                selected.add(node_id)
    if selected:
        return selected
    fallback_nodes = select_repair_nodes(
        condition_name,
        relation_labels,
        nodes,
        artifacts,
        top_k=top_k,
    )
    return {
        str(getattr(node, "node_id", "") or "")
        for node in fallback_nodes
        if getattr(node, "node_id", "")
    }


def _apply_curated_repairs(artifacts, repair_nodes: list[Any], *, top_k: int) -> tuple[Any, list[dict[str, Any]]]:
    state = _graph_state_from_artifacts(artifacts)
    curation_report: list[dict[str, Any]] = []

    for subject, relation_label in CURATED_RELATION_RESETS:
        _remove_relation_family(state, subject, relation_label)

    _rebuild_state_graph(state)
    working_artifacts = _graph_artifacts_from_state(state)

    for spec in CURATED_PAYLOADS:
        chunk_ids = _select_curated_chunk_ids(
            repair_nodes,
            working_artifacts,
            spec["condition"],
            spec["relations"],
            spec["source_hints"],
            top_k=top_k,
        )
        applied = _apply_extracted_payload_to_state(
            state,
            spec["payload"],
            chunk_ids=chunk_ids,
            target_condition=spec["condition"],
            allowed_relations=set(spec["relations"]),
        )
        curation_report.append(
            {
                "condition": spec["condition"],
                "relations": spec["relations"],
                "chunk_ids": sorted(chunk_ids),
                **applied,
            }
        )
        _rebuild_state_graph(state)
        working_artifacts = _graph_artifacts_from_state(state)

    return _graph_artifacts_from_state(state), curation_report


def _node_checkpoint_exists(checkpoint_dir: str | Path) -> bool:
    return (Path(checkpoint_dir) / "enriched_nodes.pkl").exists()


def load_clinical_repair_nodes(
    clinical_node_dir: str | Path,
    supplemental_node_dir: str | Path | None = None,
) -> tuple[list[Any], list[str]]:
    primary_dir = Path(clinical_node_dir)
    clinical_nodes = load_enriched_nodes_checkpoint(str(primary_dir))
    node_groups = [clinical_nodes]
    node_sources = [str(primary_dir)]

    if supplemental_node_dir:
        supplemental_dir = Path(supplemental_node_dir)
        if (
            supplemental_dir.resolve() != primary_dir.resolve()
            and _node_checkpoint_exists(supplemental_dir)
        ):
            node_groups.append(load_enriched_nodes_checkpoint(str(supplemental_dir)))
            node_sources.append(str(supplemental_dir))

    return _dedupe_nodes(*node_groups), node_sources


def _checkpoint_meta_summary(checkpoint_dir: str | Path, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_dir": str(Path(checkpoint_dir)),
        "checkpoint_version": meta.get("checkpoint_version"),
        "generated_at": meta.get("generated_at"),
        "prompt_version": meta.get("prompt_version"),
        "repair_prompt_version": meta.get("repair_prompt_version"),
        "model_name": meta.get("model_name"),
        "repair_model_name": meta.get("repair_model_name"),
        "node_count": meta.get("node_count"),
        "nodes_hash": meta.get("nodes_hash"),
    }


def _repair_lineage(source_graph_dir: str | Path, source_meta: dict[str, Any]) -> list[dict[str, Any]]:
    lineage = list(source_meta.get("lineage", []))
    source_summary = _checkpoint_meta_summary(source_graph_dir, source_meta)
    if not lineage or lineage[-1].get("checkpoint_dir") != source_summary["checkpoint_dir"]:
        lineage.append(source_summary)
    return lineage


def build_clinical_repair_metadata(
    *,
    source_graph_dir: str | Path,
    source_meta: dict[str, Any],
    repair_nodes: list[Any],
    repair_node_sources: list[str],
    repair_llm: Any,
    repair_report: dict[str, Any],
    curation_report: list[dict[str, Any]],
    cleanup_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_dir": str(Path(source_graph_dir)),
        "source_checkpoint": _checkpoint_meta_summary(source_graph_dir, source_meta),
        "lineage": _repair_lineage(source_graph_dir, source_meta),
        "repair_targets": CLINICAL_REPAIR_TARGETS,
        "repair_report": repair_report,
        "curation_report": curation_report,
        "cleanup_report": cleanup_report,
        "prompt_version": source_meta.get("prompt_version"),
        "repair_prompt_version": "v1",
        "nodes_hash": source_meta.get("nodes_hash"),
        "node_count": source_meta.get("node_count"),
        "model_name": source_meta.get("model_name"),
        "repair_model_name": getattr(getattr(repair_llm, "metadata", None), "model_name", None)
        or getattr(repair_llm, "model", None)
        or "unknown-model",
        "repair_node_count": len(repair_nodes),
        "repair_node_sources": repair_node_sources,
    }


def repair_clinical_graph_checkpoint(
    source_graph_dir: str | Path,
    target_graph_dir: str | Path,
    *,
    clinical_node_dir: str | Path,
    supplemental_node_dir: str | Path | None = None,
    llm_client: Any,
    top_k: int = 8,
    progress_every: int = 1,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    repair_nodes, repair_node_sources = load_clinical_repair_nodes(
        clinical_node_dir,
        supplemental_node_dir=supplemental_node_dir,
    )
    artifacts, source_meta = load_graph_checkpoint(source_graph_dir)
    repaired, repair_report = repair_graph_artifacts(
        artifacts,
        repair_nodes,
        llm_client,
        repair_targets=CLINICAL_REPAIR_TARGETS,
        top_k=top_k,
        progress_every=progress_every,
    )
    curated, curation_report = _apply_curated_repairs(
        repaired,
        repair_nodes,
        top_k=top_k,
    )
    cleaned, cleanup_report = clean_graph_artifacts(curated)
    cleaned = refresh_chunk_citation_map(cleaned, repair_nodes)

    meta = build_clinical_repair_metadata(
        source_graph_dir=source_graph_dir,
        source_meta=source_meta,
        repair_nodes=repair_nodes,
        repair_node_sources=repair_node_sources,
        repair_llm=llm_client,
        repair_report=repair_report,
        curation_report=curation_report,
        cleanup_report=cleanup_report,
    )
    save_graph_checkpoint(target_graph_dir, cleaned, meta=meta)

    report = {
        "repair_report": repair_report,
        "curation_report": curation_report,
        "cleanup_report": cleanup_report,
        "repair_node_count": len(repair_nodes),
        "repair_node_sources": repair_node_sources,
    }
    return cleaned, meta, report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair and clean the clinical GraphRAG checkpoint.")
    parser.add_argument("--source-graph-dir", default="./checkpoints/clinical_graph_clean")
    parser.add_argument("--target-graph-dir", default="./checkpoints/clinical_graph_clean")
    parser.add_argument("--clinical-node-dir", default="./checkpoints/clinical_pipeline")
    parser.add_argument(
        "--supplemental-node-dir",
        "--dsm-node-dir",
        dest="supplemental_node_dir",
        default="./checkpoints/pipeline",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    load_dotenv(Path(".env"), override=True)

    if hasattr(os.sys.stdout, "reconfigure"):
        os.sys.stdout.reconfigure(line_buffering=True)

    repair_llm = _maybe_build_openai_llm("PIPELINE_REPAIR", "gpt-4o-mini", 900)
    if repair_llm is None:
        raise RuntimeError("Repair requires OPENAI_API_KEY to be configured.")

    cleaned, _meta, report = repair_clinical_graph_checkpoint(
        args.source_graph_dir,
        args.target_graph_dir,
        clinical_node_dir=args.clinical_node_dir,
        supplemental_node_dir=args.supplemental_node_dir,
        llm_client=repair_llm,
        top_k=args.top_k,
        progress_every=args.progress_every,
    )

    print(f"Saved repaired clinical graph to {args.target_graph_dir}")
    print(f"Entities: {cleaned.entity_count}")
    print(f"Relations: {cleaned.relation_count}")
    print(f"Repair nodes: {report['repair_node_count']} from {report['repair_node_sources']}")
    print(f"Repair report: {report['repair_report']}")
    print(f"Curation report: {report['curation_report']}")
    print(f"Cleanup report: {report['cleanup_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
