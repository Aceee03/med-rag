from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from llama_index.core.graph_stores.types import EntityNode, Relation

import fastapi_app
import pipeline
from graphrag_pipeline import (
    GraphArtifacts,
    build_relation_index,
    graph_checkpoint_exists,
    hybrid_query,
    load_graph_checkpoint,
    load_or_build_communities,
    load_or_build_community_summaries,
    merge_graph_artifacts,
)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRepairLLM:
    model = "fake-repair-model"

    def chat(self, messages) -> _FakeResponse:  # pragma: no cover - exercised through pipeline flow
        return _FakeResponse('{"entities": [], "relations": []}')


class _CheckpointBackedService:
    def __init__(self, checkpoint_dir: str) -> None:
        self.graph_sync_source_dir = checkpoint_dir
        self.graph_checkpoint_dir = checkpoint_dir
        self.community_checkpoint_dir = checkpoint_dir
        self.session_questions: dict[str, list[str]] = {}
        self.graph_backend = None
        self.query_backend_name = "neo4j_live"
        self._neo4j_driver = None
        self.answer_llm = None
        self.artifacts, source_meta = load_graph_checkpoint(checkpoint_dir)
        self.communities = load_or_build_communities(
            self.artifacts,
            checkpoint_dir=checkpoint_dir,
            force_rebuild=False,
        )
        self.summaries = load_or_build_community_summaries(
            self.communities,
            self.artifacts,
            checkpoint_dir=checkpoint_dir,
            llm_client=None,
            force_rebuild=False,
        )
        self.graph_meta = {
            **source_meta,
            "source": "neo4j",
            "runtime_mode": "neo4j_only",
            "graph_sync_source_dir": checkpoint_dir,
        }

    def query(
        self,
        question: str,
        *,
        answer_with_llm: bool = False,
        session_id: str | None = None,
    ) -> dict[str, object]:
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
            self.session_questions[normalized_session_id] = [*history, question][-6:]
        return result

    def close(self) -> None:
        return None


class ClinicalApiEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fastapi_app.get_service.cache_clear()
        cls.service = _CheckpointBackedService("./checkpoints/clinical_dsm_merged")
        cls.get_service_patcher = patch.object(fastapi_app, "get_service", return_value=cls.service)
        cls.get_service_patcher.start()
        cls.client = TestClient(fastapi_app.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.get_service_patcher.stop()
        fastapi_app.get_service.cache_clear()

    def test_health_and_meta_routes(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        health_payload = health.json()
        self.assertEqual(health_payload["status"], "ok")
        self.assertGreater(health_payload["entity_count"], 0)
        self.assertGreater(health_payload["relation_count"], 0)

        meta = self.client.get("/meta")
        self.assertEqual(meta.status_code, 200)
        meta_payload = meta.json()
        self.assertIn("graph_meta", meta_payload)
        self.assertGreater(meta_payload["community_count"], 0)
        self.assertGreater(meta_payload["summary_count"], 0)

    def test_query_route_returns_repaired_answer(self) -> None:
        response = self.client.post(
            "/query",
            json={
                "question": "What is the prevalence of generalized anxiety disorder?",
                "answer_with_llm": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["out_of_scope"])
        self.assertEqual(payload["query_type"], "prevalence")
        self.assertIn("2.9% among adults", payload["answer"])
        self.assertTrue(payload["citations"])
        self.assertTrue(payload["source_chunks"])
        self.assertTrue(any(chunk["text"] for chunk in payload["source_chunks"]))

    def test_query_route_accepts_natural_human_symptom_question(self) -> None:
        response = self.client.post(
            "/query",
            json={
                "question": "I feel tired and lost my appetite to eat. What could this possibly be ?",
                "answer_with_llm": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["out_of_scope"])
        self.assertEqual(payload["query_type"], "reverse_symptom")
        self.assertIn("Possible matching conditions", payload["answer"])
        self.assertIn("FATIGUE", payload["answer"])
        self.assertTrue(
            "LOSS OF APPETITE" in payload["answer"] or "POOR APPETITE" in payload["answer"]
        )

    def test_query_route_handles_small_condition_typos(self) -> None:
        response = self.client.post(
            "/query",
            json={
                "question": "what is depresion",
                "answer_with_llm": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["out_of_scope"])
        self.assertIn(payload["query_type"], {"forward_lookup", "general"})
        self.assertIn("DEPRESSION", payload["answer"])

    def test_query_route_uses_session_context_for_follow_up_symptoms(self) -> None:
        session_id = "follow-up-symptom-test"
        first = self.client.post(
            "/query",
            json={
                "question": "I feel tired all the time.",
                "session_id": session_id,
                "answer_with_llm": False,
            },
        )
        self.assertEqual(first.status_code, 200)
        first_payload = first.json()
        self.assertFalse(first_payload["out_of_scope"])

        second = self.client.post(
            "/query",
            json={
                "question": "and I lost my appetite to eat lately.",
                "session_id": session_id,
                "answer_with_llm": False,
            },
        )
        self.assertEqual(second.status_code, 200)
        second_payload = second.json()
        self.assertFalse(second_payload["out_of_scope"])
        self.assertEqual(second_payload["query_type"], "reverse_symptom")
        self.assertIn("FATIGUE", second_payload["answer"])
        self.assertTrue(
            "LOSS OF APPETITE" in second_payload["answer"] or "POOR APPETITE" in second_payload["answer"]
        )


class PipelineClinicalRepairEndToEndTests(unittest.TestCase):
    def test_pipeline_can_rebuild_clinical_repair_checkpoint_in_one_command(self) -> None:
        fake_repair_llm = _FakeRepairLLM()
        target_graph_dir = Path("checkpoints") / f"tmp_clinical_graph_repaired_{uuid4().hex}"
        shutil.rmtree(target_graph_dir, ignore_errors=True)

        with patch.object(
            pipeline,
            "_maybe_build_openai_llm",
            return_value=fake_repair_llm,
        ):
            try:
                exit_code = pipeline.main(
                    [
                        "--graph-checkpoint-dir",
                        "./checkpoints/clinical_graph_clean",
                        "--community-checkpoint-dir",
                        str(target_graph_dir),
                        "--repair-clinical-gaps",
                        "--clinical-repair-graph-checkpoint-dir",
                        str(target_graph_dir),
                        "--clinical-repair-node-dir",
                        "./checkpoints/clinical_pipeline",
                        "--clinical-repair-supplemental-node-dir",
                        "./checkpoints/pipeline",
                        "--progress-every",
                        "1",
                    ]
                )

                self.assertEqual(exit_code, 0)
                self.assertTrue(graph_checkpoint_exists(target_graph_dir))

                artifacts, meta = load_graph_checkpoint(target_graph_dir)
                self.assertIn("curation_report", meta)
                self.assertIn("lineage", meta)
                self.assertNotIn("source_meta", meta)
                normalized_sources = {Path(source).as_posix() for source in meta["repair_node_sources"]}
                self.assertIn("checkpoints/clinical_pipeline", normalized_sources)

                result = hybrid_query(
                    "What treatments are available for autism spectrum disorder?",
                    artifacts,
                    community_summaries={},
                )
                self.assertIn("BEHAVIORAL THERAPY", result["answer"])
                self.assertIn("PSYCHOSOCIAL INTERVENTIONS", result["answer"])
            finally:
                shutil.rmtree(target_graph_dir, ignore_errors=True)


class Neo4jSyncBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts, _ = load_graph_checkpoint("./checkpoints/clinical_graph_clean")

    def test_sync_to_neo4j_calls_helpers_and_closes_driver(self) -> None:
        fake_driver = Mock()
        with patch("neo4j_helpers.create_driver", return_value=fake_driver), patch(
            "neo4j_helpers.test_connection",
            return_value=True,
        ) as test_connection, patch("neo4j_helpers.clear_neo4j_graph") as clear_graph, patch(
            "neo4j_helpers.create_neo4j_indexes"
        ) as create_indexes, patch("neo4j_helpers.write_graph_to_neo4j") as write_graph:
            pipeline._sync_to_neo4j(self.artifacts)

        test_connection.assert_called_once_with(fake_driver)
        clear_graph.assert_called_once_with(fake_driver)
        create_indexes.assert_called_once_with(fake_driver)
        write_args = write_graph.call_args.args
        self.assertIs(write_args[0], fake_driver)
        self.assertIs(write_args[1], self.artifacts.custom_entities)
        self.assertEqual(write_args[3], self.artifacts.relation_metadata)
        self.assertEqual(write_args[4], self.artifacts.entity_sources)
        self.assertEqual(write_args[5], self.artifacts.chunk_citation_map)
        self.assertEqual(write_args[6], self.artifacts.chunk_payload_map)
        fake_driver.close.assert_called_once()

    def test_sync_to_neo4j_raises_on_failed_connection_and_still_closes_driver(self) -> None:
        fake_driver = Mock()
        with patch("neo4j_helpers.create_driver", return_value=fake_driver), patch(
            "neo4j_helpers.test_connection",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "Neo4j connection test failed"):
                pipeline._sync_to_neo4j(self.artifacts)

        fake_driver.close.assert_called_once()


def _build_test_artifacts(
    entity_specs: list[tuple[str, str, str, set[str]]],
    relation_specs: list[tuple[str, str, str, str, int, set[str]]],
    *,
    chunk_citation_map: dict[str, str] | None = None,
    chunk_payload_map: dict[str, dict[str, str]] | None = None,
) -> GraphArtifacts:
    custom_entities: dict[str, EntityNode] = {}
    entity_descriptions: dict[str, str] = {}
    entity_sources: dict[str, set[str]] = {}

    for name, label, description, sources in entity_specs:
        node = EntityNode(
            name=name,
            label=label,
            properties={"description": description, "sources": sorted(sources)},
        )
        custom_entities[name] = node
        entity_descriptions[name] = description
        entity_sources[name] = set(sources)

    entity_id_to_node = {entity.id: entity for entity in custom_entities.values()}
    custom_relations: list[Relation] = []
    relation_metadata: dict[tuple[str, str, str], dict[str, object]] = {}
    for subject, relation_label, target, description, strength, sources in relation_specs:
        custom_relations.append(
            Relation(
                source_id=custom_entities[subject].id,
                target_id=custom_entities[target].id,
                label=relation_label,
            )
        )
        relation_metadata[(subject, relation_label, target)] = {
            "description": description,
            "strength": strength,
            "sources": set(sources),
        }

    return GraphArtifacts(
        custom_entities=custom_entities,
        entity_id_to_node=entity_id_to_node,
        custom_relations=custom_relations,
        relation_metadata=relation_metadata,
        entity_descriptions=entity_descriptions,
        entity_sources=entity_sources,
        relation_index=build_relation_index(custom_entities, custom_relations, entity_id_to_node),
        chunk_citation_map=chunk_citation_map or {},
        chunk_payload_map=chunk_payload_map or {},
    )


class _FakeGraphBackend:
    def __init__(self) -> None:
        self.forward_calls: list[tuple[str, set[str] | None]] = []
        self.entity_calls: list[tuple[str, set[str] | None]] = []
        self.reverse_calls: list[list[str]] = []
        self.text_calls: list[str] = []

    def forward_lookup(
        self,
        condition_name: str,
        *,
        relation_filter: set[str] | None = None,
    ) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
        self.forward_calls.append((condition_name, relation_filter))
        return (
            {
                "HAS_SYMPTOM": [
                    {
                        "name": "FATIGUE",
                        "type": "SYMPTOM",
                        "strength": 9,
                        "description": "Persistent low energy.",
                    }
                ]
            },
            {"chunk-1"},
        )

    def entity_relation_lookup(
        self,
        entity_name: str,
        *,
        relation_filter: set[str] | None = None,
    ) -> tuple[dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]], set[str]]:
        self.entity_calls.append((entity_name, relation_filter))
        return {}, {}, set()

    def reverse_symptom_lookup(self, symptom_names: list[str]) -> tuple[list[tuple[str, list[str]]], set[str]]:
        self.reverse_calls.append(symptom_names)
        return [("DEPRESSION", ["FATIGUE"])], {"chunk-1"}

    def text_search_entities(self, question: str, *, limit: int = 10) -> list[dict[str, object]]:
        self.text_calls.append(question)
        return [
            {
                "name": "DEPRESSION",
                "type": "CONDITION",
                "description": "Mood disorder.",
                "score": 4,
                "sources": ["chunk-1"],
            }
        ]


class HybridQueryBackendTests(unittest.TestCase):
    def test_hybrid_query_uses_live_graph_backend_for_forward_lookup(self) -> None:
        artifacts = _build_test_artifacts(
            [
                ("DEPRESSION", "CONDITION", "Mood disorder.", {"chunk-1"}),
                ("FATIGUE", "SYMPTOM", "Low energy.", {"chunk-1"}),
            ],
            [],
            chunk_citation_map={"chunk-1": "Test Citation"},
        )
        backend = _FakeGraphBackend()

        result = hybrid_query(
            "What are the symptoms of depression?",
            artifacts,
            community_summaries={},
            graph_backend=backend,
        )

        self.assertEqual(result["query_type"], "symptoms")
        self.assertIn("FATIGUE", result["answer"])
        self.assertEqual(
            backend.forward_calls,
            [
                ("DEPRESSION", {"HAS_SYMPTOM"}),
                ("DEPRESSION", {"HAS_DIAGNOSTIC_CRITERION"}),
            ],
        )
        self.assertEqual(backend.text_calls, [])
        self.assertEqual(result["citations"], ["Test Citation"])


class GraphMergeBehaviorTests(unittest.TestCase):
    def test_merge_keeps_primary_label_on_conflict_and_preserves_valid_relations(self) -> None:
        primary = _build_test_artifacts(
            [
                ("POST TRAUMATIC STRESS DISORDER", "CONDITION", "Primary PTSD description.", {"p-1"}),
                ("NIGHTMARE", "SYMPTOM", "Primary symptom description.", {"p-1"}),
            ],
            [
                (
                    "POST TRAUMATIC STRESS DISORDER",
                    "HAS_SYMPTOM",
                    "NIGHTMARE",
                    "Primary symptom edge.",
                    5,
                    {"p-1"},
                )
            ],
            chunk_citation_map={"p-1": "PRIMARY: PTSD"},
        )
        secondary = _build_test_artifacts(
            [
                ("POST TRAUMATIC STRESS DISORDER", "CONDITION", "Secondary PTSD description.", {"s-1"}),
                ("NIGHTMARE", "CONDITION", "Secondary nightmare condition.", {"s-1"}),
                ("TRAUMA", "RISK_FACTOR", "Risk factor from secondary graph.", {"s-1"}),
                ("PANIC DISORDER", "CONDITION", "Differential diagnosis from secondary graph.", {"s-1"}),
            ],
            [
                (
                    "POST TRAUMATIC STRESS DISORDER",
                    "HAS_SYMPTOM",
                    "NIGHTMARE",
                    "Secondary symptom edge.",
                    9,
                    {"s-1"},
                ),
                (
                    "NIGHTMARE",
                    "HAS_RISK_FACTOR",
                    "TRAUMA",
                    "Invalid once NIGHTMARE stays a symptom.",
                    7,
                    {"s-1"},
                ),
                (
                    "POST TRAUMATIC STRESS DISORDER",
                    "DIFFERENTIAL_DIAGNOSIS",
                    "PANIC DISORDER",
                    "Valid secondary relation.",
                    8,
                    {"s-1"},
                ),
            ],
            chunk_citation_map={"s-1": "SECONDARY: PTSD"},
        )

        merged, report = merge_graph_artifacts(primary, secondary)

        self.assertEqual(merged.custom_entities["NIGHTMARE"].label, "SYMPTOM")
        self.assertIn(("POST TRAUMATIC STRESS DISORDER", "HAS_SYMPTOM", "NIGHTMARE"), merged.relation_metadata)
        symptom_meta = merged.relation_metadata[("POST TRAUMATIC STRESS DISORDER", "HAS_SYMPTOM", "NIGHTMARE")]
        self.assertEqual(symptom_meta["strength"], 9)
        self.assertEqual(symptom_meta["sources"], {"p-1", "s-1"})
        self.assertIn(
            ("POST TRAUMATIC STRESS DISORDER", "DIFFERENTIAL_DIAGNOSIS", "PANIC DISORDER"),
            merged.relation_metadata,
        )
        self.assertNotIn(("NIGHTMARE", "HAS_RISK_FACTOR", "TRAUMA"), merged.relation_metadata)
        self.assertEqual(report["type_conflicts"], 1)
        self.assertEqual(report["type_conflict_entities"][0]["name"], "NIGHTMARE")


if __name__ == "__main__":
    unittest.main()
