"""Neo4j connection + query helpers for the notebook pipeline."""

import os
import re
import time
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

RELATION_LABEL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
TEXT_QUERY_STOPWORDS = {
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
    "symptom",
    "symptoms",
    "condition",
    "conditions",
    "disorder",
    "disorders",
    "could",
    "might",
    "would",
    "used",
}


def _require_env(name: str) -> str:
    """Read a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _neo4j_uri() -> str:
    return _require_env("NEO4J_URI")


def _neo4j_username() -> str:
    return _require_env("NEO4J_USERNAME")


def _neo4j_password() -> str:
    return _require_env("NEO4J_PASSWORD")


def _neo4j_database() -> str:
    return _require_env("NEO4J_DATABASE")


def create_driver():
    """Create and return a Neo4j driver."""
    return GraphDatabase.driver(
        _neo4j_uri(),
        auth=(_neo4j_username(), _neo4j_password()),
    )


def test_connection(driver) -> bool:
    """Test Neo4j connectivity and print status."""
    try:
        def _probe() -> str:
            with driver.session(database=_neo4j_database()) as session:
                result = session.run("RETURN 'connected' AS status")
                return result.single()["status"]

        status = _run_with_retry(_probe, operation_name="Neo4j connection test")
        print(f"Neo4j connection: {status}")
        print(f"URI: {_neo4j_uri()}")
        print(f"Database: {_neo4j_database()}")
        return True
    except Exception as exc:
        print(f"Neo4j connection failed: {exc}")
        return False


def _normalize_relation_filter(relation_filter: set[str] | list[str] | None) -> list[str] | None:
    if not relation_filter:
        return None
    normalized = sorted({str(item).strip().upper() for item in relation_filter if str(item).strip()})
    return normalized or None


def _question_keywords(question: str) -> list[str]:
    ordered_terms: list[str] = []
    seen: set[str] = set()
    for term in re.findall(r"[a-z0-9]+", question.lower()):
        if len(term) <= 2 or term in TEXT_QUERY_STOPWORDS or term in seen:
            continue
        seen.add(term)
        ordered_terms.append(term)
    return ordered_terms


def _update_chunk_citation_map(
    chunk_citation_map: dict[str, str],
    sources: list[str] | None,
    citations: list[str] | None,
) -> None:
    for source, citation in zip_longest(sources or [], citations or [], fillvalue=None):
        if not source:
            continue
        chunk_citation_map.setdefault(str(source), str(citation or source))


def _chunked(rows: list[dict[str, Any]], chunk_size: int = 500):
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


def _run_with_retry(
    operation,
    *,
    attempts: int = 4,
    initial_delay_seconds: float = 1.0,
    backoff: float = 2.0,
    operation_name: str = "Neo4j operation",
):
    delay_seconds = initial_delay_seconds
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            print(
                f"{operation_name} failed on attempt {attempt}/{attempts}: {exc}. "
                f"Retrying in {delay_seconds:.1f}s..."
            )
            time.sleep(delay_seconds)
            delay_seconds *= backoff
    raise last_exc


def write_graph_to_neo4j(
    driver,
    custom_entities,
    seen_relations,
    relation_metadata,
    entity_sources,
    chunk_citation_map=None,
    chunk_payload_map=None,
):
    """Write the extracted graph to Neo4j using MERGE semantics."""
    if chunk_citation_map is None:
        chunk_citation_map = {}
    if chunk_payload_map is None:
        chunk_payload_map = {}

    def _write_entity_batch(tx, rows):
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (n:Entity {name: row.name})
            SET n.type = row.type,
                n.description = row.description,
                n.sources = row.sources,
                n.citations = row.citations
            """,
            rows=rows,
        )

    def _write_relation_batch(tx, relation_label: str, rows):
        query = f"""
            UNWIND $rows AS row
            MATCH (a:Entity {{name: row.subject}})
            MATCH (b:Entity {{name: row.target}})
            MERGE (a)-[r:{relation_label}]->(b)
            SET r.strength = row.strength,
                r.description = row.description,
                r.sources = row.sources,
                r.citations = row.citations
        """
        tx.run(query, rows=rows)

    def _write_chunk_batch(tx, rows):
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (c:Chunk {id: row.id})
            SET c.citation = row.citation,
                c.text = row.text,
                c.section_title = row.section_title,
                c.context_tag = row.context_tag,
                c.source_label = row.source_label,
                c.file_path = row.file_path,
                c.header_path = row.header_path
            """,
            rows=rows,
        )

    entity_rows: list[dict[str, Any]] = []
    relation_rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunk_rows: list[dict[str, Any]] = []

    for canonical_name, entity in custom_entities.items():
        sources = sorted(entity_sources.get(canonical_name, set()))
        entity_rows.append(
            {
                "name": canonical_name,
                "type": entity.label,
                "description": entity.properties.get("description", ""),
                "sources": sources,
                "citations": [chunk_citation_map.get(source, source) for source in sources],
            }
        )

    for relation_key in seen_relations:
        if not isinstance(relation_key, tuple) or len(relation_key) != 3:
            continue
        subject, relation_label, target = relation_key
        if not RELATION_LABEL_PATTERN.fullmatch(str(relation_label)):
            raise ValueError(f"Invalid Neo4j relation label: {relation_label!r}")
        metadata = relation_metadata.get((subject, relation_label, target), {})
        relation_sources = sorted(metadata.get("sources", set()))
        relation_rows_by_label[relation_label].append(
            {
                "subject": subject,
                "target": target,
                "strength": metadata.get("strength", 5),
                "description": metadata.get("description", ""),
                "sources": relation_sources,
                "citations": [chunk_citation_map.get(source, source) for source in relation_sources],
            }
        )

    for chunk_id in sorted(set(chunk_citation_map) | set(chunk_payload_map)):
        payload = dict(chunk_payload_map.get(chunk_id, {}))
        chunk_rows.append(
            {
                "id": str(chunk_id),
                "citation": str(payload.get("citation") or chunk_citation_map.get(chunk_id, chunk_id)),
                "text": str(payload.get("text", "") or ""),
                "section_title": str(payload.get("section_title", "") or ""),
                "context_tag": str(payload.get("context_tag", "") or ""),
                "source_label": str(payload.get("source_label", "") or ""),
                "file_path": str(payload.get("file_path", "") or ""),
                "header_path": str(payload.get("header_path", "") or ""),
            }
        )

    with driver.session(database=_neo4j_database()) as session:
        for batch in _chunked(entity_rows):
            session.execute_write(_write_entity_batch, batch)
        for relation_label, rows in sorted(relation_rows_by_label.items()):
            for batch in _chunked(rows):
                session.execute_write(_write_relation_batch, relation_label, batch)
        for batch in _chunked(chunk_rows):
            session.execute_write(_write_chunk_batch, batch)


def clear_neo4j_graph(driver):
    """Delete all nodes and relationships from the configured Neo4j database."""
    with driver.session(database=_neo4j_database()) as session:
        session.run("MATCH (n) DETACH DELETE n")


def create_neo4j_indexes(driver):
    """Create the indexes and constraints used by the GraphRAG pipeline."""
    with driver.session(database=_neo4j_database()) as session:
        session.run(
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE n.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        )
        session.run(
            """
            CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
            FOR (n:Entity) ON EACH [n.name, n.description]
            """
        )
        session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)")


def get_neo4j_graph_stats(driver) -> dict[str, int]:
    """Return basic graph counts for the configured Neo4j database."""
    def _query_stats() -> dict[str, int]:
        with driver.session(database=_neo4j_database()) as session:
            entity_count = session.run(
                """
                MATCH (n:Entity)
                RETURN count(n) AS count
                """
            ).single()["count"]
            relation_count = session.run(
                """
                MATCH (:Entity)-[r]->(:Entity)
                RETURN count(r) AS count
                """
            ).single()["count"]
            chunk_count = session.run(
                """
                MATCH (c:Chunk)
                RETURN count(c) AS count
                """
            ).single()["count"]
        return {
            "entity_count": int(entity_count),
            "relation_count": int(relation_count),
            "chunk_count": int(chunk_count),
        }

    return _run_with_retry(_query_stats, operation_name="Neo4j graph stats")


def neo4j_forward_lookup(driver, condition_name: str) -> dict:
    """
    What are the symptoms and treatments of X?
    Returns dict grouped by relation type.
    """
    with driver.session(database=_neo4j_database()) as session:
        result = session.run(
            """
            MATCH (c:Entity {name: $name})-[r]->(target:Entity)
            RETURN type(r) AS relation,
                   target.name AS target,
                   target.type AS target_type,
                   r.strength AS strength,
                   r.description AS description
            ORDER BY r.strength DESC
            """,
            name=condition_name.upper(),
        )
        rows = result.data()

    grouped = {}
    for row in rows:
        rel = row["relation"]
        grouped.setdefault(rel, []).append(
            {
                "name": row["target"],
                "type": row["target_type"],
                "strength": row["strength"],
                "description": row["description"],
            }
        )
    return grouped


def neo4j_reverse_lookup(driver, symptom_names: list) -> list:
    """
    Which conditions have these symptoms? (differential diagnosis)
    Returns list of conditions ranked by match count.
    """
    symptom_names = [s.upper() for s in symptom_names]
    with driver.session(database=_neo4j_database()) as session:
        result = session.run(
            """
            MATCH (c:Entity)-[:HAS_SYMPTOM]->(s:Entity)
            WHERE s.name IN $symptoms
              AND c.type = 'CONDITION'
            RETURN c.name AS condition,
                   c.description AS description,
                   collect(s.name) AS matched_symptoms,
                   count(s) AS match_count
            ORDER BY match_count DESC
            """,
            symptoms=symptom_names,
        )
        return result.data()


def neo4j_comparison(driver, condition_a: str, condition_b: str) -> dict:
    """
    What do two conditions share vs differ on?
    Returns dict with 'shared' and 'unique_a' and 'unique_b' keys.
    """
    condition_a = condition_a.upper()
    condition_b = condition_b.upper()

    result_dict = {"shared": {}, "unique_a": {}, "unique_b": {}}

    with driver.session(database=_neo4j_database()) as session:
        # Find shared connections
        shared_result = session.run(
            """
            MATCH (ca:Entity {name: $a})-[ra]->(shared:Entity)<-[rb]-(cb:Entity {name: $b})
            WHERE type(ra) = type(rb)
            RETURN type(ra) AS relation, shared.name AS entity, shared.type AS type
            """,
            a=condition_a,
            b=condition_b,
        )
        for row in shared_result.data():
            rel = row["relation"]
            result_dict["shared"].setdefault(rel, set()).add(row["entity"])

        # Find unique to A
        unique_a_result = session.run(
            """
            MATCH (ca:Entity {name: $a})-[r]->(unique_a:Entity)
            WHERE NOT EXISTS {
                MATCH (cb:Entity {name: $b})-[]->(unique_a)
            }
            RETURN type(r) AS relation, unique_a.name AS entity, unique_a.type AS type
            """,
            a=condition_a,
            b=condition_b,
        )
        for row in unique_a_result.data():
            rel = row["relation"]
            result_dict["unique_a"].setdefault(rel, set()).add(row["entity"])

        # Find unique to B
        unique_b_result = session.run(
            """
            MATCH (cb:Entity {name: $b})-[r]->(unique_b:Entity)
            WHERE NOT EXISTS {
                MATCH (ca:Entity {name: $a})-[]->(unique_b)
            }
            RETURN type(r) AS relation, unique_b.name AS entity, unique_b.type AS type
            """,
            a=condition_a,
            b=condition_b,
        )
        for row in unique_b_result.data():
            rel = row["relation"]
            result_dict["unique_b"].setdefault(rel, set()).add(row["entity"])

    return result_dict


def neo4j_text_query(driver, keywords: list, limit: int = 20) -> list:
    """
    Full-text keyword search across all node names and descriptions.
    Falls back to CONTAINS if full-text index doesn't exist.
    """
    normalized_keywords = [str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()]
    if not normalized_keywords:
        return []

    try:
        query_string = " OR ".join(normalized_keywords)
        def _fulltext_query():
            with driver.session(database=_neo4j_database()) as session:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('entity_search', $q)
                    YIELD node, score
                    RETURN node.name AS name, node.type AS type,
                           node.description AS description,
                           coalesce(node.sources, []) AS sources,
                           score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    q=query_string,
                    limit=limit,
                )
                return result.data()

        return _run_with_retry(_fulltext_query, operation_name="Neo4j full-text search")
    except Exception:
        # Fallback to simple CONTAINS search
        def _fallback_query():
            with driver.session(database=_neo4j_database()) as session:
                result = session.run(
                    """
                    MATCH (n:Entity)
                    WITH n, [
                        keyword IN $keywords
                        WHERE toLower(n.name) CONTAINS keyword
                           OR toLower(coalesce(n.description, '')) CONTAINS keyword
                    ] AS matches
                    WHERE size(matches) > 0
                    RETURN n.name AS name, n.type AS type,
                           n.description AS description,
                           coalesce(n.sources, []) AS sources,
                           size(matches) AS score
                    ORDER BY score DESC, n.name
                    LIMIT $limit
                    """,
                    keywords=normalized_keywords,
                    limit=limit,
                )
                return result.data()

        return _run_with_retry(_fallback_query, operation_name="Neo4j fallback text search")


def neo4j_get_all_entities(driver) -> list:
    """Get all entities from Neo4j."""
    with driver.session(database=_neo4j_database()) as session:
        result = session.run(
            """
            MATCH (n:Entity)
            RETURN n.name AS name, n.type AS type, n.description AS description
            ORDER BY n.name
            """
        )
        return result.data()


def neo4j_get_all_relations(driver) -> list:
    """Get all relations from Neo4j."""
    with driver.session(database=_neo4j_database()) as session:
        result = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            RETURN a.name AS source, type(r) AS relation, b.name AS target,
                   r.strength AS strength, r.description AS description
            ORDER BY a.name, type(r)
            """
        )
        return result.data()


def neo4j_get_entity_neighbors(driver, entity_name: str) -> dict:
    """Get all neighbors (incoming and outgoing) of an entity."""
    entity_name = entity_name.upper()

    with driver.session(database=_neo4j_database()) as session:
        # Outgoing
        out_result = session.run(
            """
            MATCH (n:Entity {name: $name})-[r]->(target:Entity)
            RETURN type(r) AS relation, target.name AS neighbor,
                   target.type AS type, 'outgoing' AS direction
            """,
            name=entity_name,
        )
        outgoing = out_result.data()

        # Incoming
        in_result = session.run(
            """
            MATCH (source:Entity)-[r]->(n:Entity {name: $name})
            RETURN type(r) AS relation, source.name AS neighbor,
                   source.type AS type, 'incoming' AS direction
            """,
            name=entity_name,
        )
        incoming = in_result.data()

    return {"outgoing": outgoing, "incoming": incoming}


class Neo4jGraphQueryBackend:
    """Live Neo4j traversal backend used by the API at query time."""

    def __init__(self, driver, database: str | None = None) -> None:
        self.driver = driver
        self.database = database or _neo4j_database()

    def _directional_lookup(
        self,
        entity_name: str,
        *,
        incoming: bool,
        relation_filter: set[str] | list[str] | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
        normalized_name = str(entity_name or "").strip().upper()
        if not normalized_name:
            return {}, set()

        normalized_filter = _normalize_relation_filter(relation_filter)
        if incoming:
            query = """
                MATCH (neighbor:Entity)-[r]->(root:Entity {name: $name})
                WHERE $relation_filter IS NULL OR type(r) IN $relation_filter
                RETURN type(r) AS relation,
                       neighbor.name AS name,
                       neighbor.type AS type,
                       coalesce(r.strength, 5) AS strength,
                       coalesce(r.description, '') AS description,
                       coalesce(r.sources, []) AS sources
                ORDER BY relation, strength DESC, name
            """
        else:
            query = """
                MATCH (root:Entity {name: $name})-[r]->(neighbor:Entity)
                WHERE $relation_filter IS NULL OR type(r) IN $relation_filter
                RETURN type(r) AS relation,
                       neighbor.name AS name,
                       neighbor.type AS type,
                       coalesce(r.strength, 5) AS strength,
                       coalesce(r.description, '') AS description,
                       coalesce(r.sources, []) AS sources
                ORDER BY relation, strength DESC, name
            """

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        citations: set[str] = set()
        def _fetch_rows():
            with self.driver.session(database=self.database) as session:
                return session.run(
                    query,
                    name=normalized_name,
                    relation_filter=normalized_filter,
                ).data()

        rows = _run_with_retry(_fetch_rows, operation_name="Neo4j directional lookup")

        for row in rows:
            grouped[row["relation"]].append(
                {
                    "name": row["name"],
                    "type": row["type"],
                    "strength": row["strength"],
                    "description": row["description"],
                }
            )
            citations.update(str(source) for source in (row.get("sources") or []) if source)

        for values in grouped.values():
            values.sort(key=lambda item: (-int(item.get("strength", 0) or 0), item["name"]))
        return dict(grouped), citations

    def forward_lookup(
        self,
        condition_name: str,
        *,
        relation_filter: set[str] | list[str] | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
        return self._directional_lookup(
            condition_name,
            incoming=False,
            relation_filter=relation_filter,
        )

    def entity_relation_lookup(
        self,
        entity_name: str,
        *,
        relation_filter: set[str] | list[str] | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], set[str]]:
        outgoing, outgoing_citations = self._directional_lookup(
            entity_name,
            incoming=False,
            relation_filter=relation_filter,
        )
        incoming, incoming_citations = self._directional_lookup(
            entity_name,
            incoming=True,
            relation_filter=relation_filter,
        )
        return outgoing, incoming, outgoing_citations | incoming_citations

    def reverse_symptom_lookup(self, symptom_names: list[str]) -> tuple[list[tuple[str, list[str]]], set[str]]:
        normalized_symptoms = sorted(
            {
                str(symptom_name).strip().upper()
                for symptom_name in symptom_names
                if str(symptom_name).strip()
            }
        )
        if not normalized_symptoms:
            return [], set()

        def _fetch_rows():
            with self.driver.session(database=self.database) as session:
                return session.run(
                    """
                    UNWIND $symptoms AS symptom_name
                    MATCH (c:Entity {type: 'CONDITION'})-[r:HAS_SYMPTOM]->(s:Entity {name: symptom_name})
                    WITH c,
                         collect(DISTINCT s.name) AS matched_symptoms,
                         collect(coalesce(r.sources, [])) AS source_lists
                    RETURN c.name AS condition,
                           matched_symptoms,
                           size(matched_symptoms) AS match_count,
                           source_lists
                    ORDER BY match_count DESC, condition
                    """,
                    symptoms=normalized_symptoms,
                ).data()

        rows = _run_with_retry(_fetch_rows, operation_name="Neo4j reverse symptom lookup")

        ranked = [
            (str(row["condition"]), sorted({str(item) for item in (row.get("matched_symptoms") or []) if item}))
            for row in rows
        ]
        citations: set[str] = set()
        for row in rows:
            for source_list in row.get("source_lists") or []:
                citations.update(str(source) for source in (source_list or []) if source)
        return ranked, citations

    def text_search_entities(self, question: str, *, limit: int = 10) -> list[dict[str, Any]]:
        return neo4j_text_query(self.driver, _question_keywords(question), limit=limit)


# ============================================================
# RESTORE FROM NEO4J
# Run this instead of loading pickles - rebuilds in-memory structures
# ============================================================

def restore_from_neo4j(driver):
    """
    Rebuild custom_entities, entity_id_to_node, custom_relations, 
    relation_metadata, and source chunk payloads from Neo4j.
    """
    from llama_index.core.graph_stores.types import EntityNode, Relation
    
    custom_entities = {}
    entity_id_to_node = {}
    entity_descriptions = {}
    entity_sources = {}
    custom_relations = []
    relation_metadata = {}
    relation_index: dict[str, dict[str, list[tuple[str, str]]]] = {}
    chunk_citation_map: dict[str, str] = {}
    chunk_payload_map: dict[str, dict[str, Any]] = {}
    
    database = _neo4j_database()

    with driver.session(database=database) as session:
        total_nodes = session.run(
            """
            MATCH (n)
            RETURN count(n) AS count
            """
        ).single()["count"]

        if total_nodes == 0:
            print(
                f"Neo4j database '{database}' is empty. "
                "Nothing to restore yet."
            )
            return (
                custom_entities,
                entity_id_to_node,
                custom_relations,
                relation_metadata,
                entity_descriptions,
                entity_sources,
                relation_index,
                chunk_citation_map,
                chunk_payload_map,
            )

        entity_node_count = session.run(
            """
            MATCH (n:Entity)
            RETURN count(n) AS count
            """
        ).single()["count"]

        if entity_node_count == 0:
            raise RuntimeError(
                "Neo4j contains data, but no :Entity nodes were found. "
                "restore_from_neo4j() expects the graph written by write_graph_to_neo4j()."
            )

        # Load nodes
        nodes = session.run(
            """
            MATCH (n:Entity)
            RETURN n.name AS name, n.type AS type, 
                   n.description AS desc, n.sources AS sources, n.citations AS citations
            """
        ).data()
        
        for row in nodes:
            name = row["name"]
            node = EntityNode(
                name=name,
                label=row["type"] or "UNKNOWN",
                properties={
                    "description": row["desc"] or "",
                    "sources": row["sources"] or []
                }
            )
            custom_entities[name] = node
            entity_id_to_node[node.id] = node
            entity_descriptions[name] = row["desc"] or ""
            entity_sources[name] = set(row["sources"] or [])
            _update_chunk_citation_map(
                chunk_citation_map,
                row.get("sources") or [],
                row.get("citations") or [],
            )
        
        # Load edges
        edges = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            RETURN a.name AS subj, type(r) AS rel, b.name AS obj,
                   r.strength AS strength, r.description AS desc,
                   r.sources AS sources, r.citations AS citations
            """
        ).data()
        
        for row in edges:
            subj, rel, obj = row["subj"], row["rel"], row["obj"]
            
            if subj in custom_entities and obj in custom_entities:
                relation = Relation(
                    source_id=custom_entities[subj].id,
                    target_id=custom_entities[obj].id,
                    label=rel
                )
                custom_relations.append(relation)
                
                relation_metadata[(subj, rel, obj)] = {
                    "description": row["desc"] or "",
                    "strength": row["strength"] or 5,
                    "sources": set(row["sources"] or [])
                }
                _update_chunk_citation_map(
                    chunk_citation_map,
                    row.get("sources") or [],
                    row.get("citations") or [],
                )

        chunk_rows = session.run(
            """
            MATCH (c:Chunk)
            RETURN c.id AS id,
                   c.citation AS citation,
                   c.text AS text,
                   c.section_title AS section_title,
                   c.context_tag AS context_tag,
                   c.source_label AS source_label,
                   c.file_path AS file_path,
                   c.header_path AS header_path
            """
        ).data()
        for row in chunk_rows:
            chunk_id = str(row.get("id") or "").strip()
            if not chunk_id:
                continue
            citation = str(row.get("citation") or chunk_citation_map.get(chunk_id, chunk_id))
            chunk_citation_map[chunk_id] = citation
            chunk_payload_map[chunk_id] = {
                "citation": citation,
                "text": str(row.get("text") or ""),
                "section_title": str(row.get("section_title") or ""),
                "context_tag": str(row.get("context_tag") or ""),
                "source_label": str(row.get("source_label") or ""),
                "file_path": str(row.get("file_path") or ""),
                "header_path": str(row.get("header_path") or ""),
            }

    # Build relation_index from loaded edges (same structure as graphrag_pipeline.build_relation_index)
    relation_index = {
        name: {"out": [], "in": []} for name in custom_entities
    }
    for row in edges:
        subj, rel, obj = row["subj"], row["rel"], row["obj"]
        if subj in custom_entities and obj in custom_entities:
            relation_index.setdefault(subj, {"out": [], "in": []})
            relation_index.setdefault(obj, {"out": [], "in": []})
            relation_index[subj]["out"].append((rel, obj))
            relation_index[obj]["in"].append((rel, subj))

    print(f"✅ Restored from Neo4j:")
    print(f"   Database:        {database}")
    print(f"   Entities:        {len(custom_entities)}")
    print(f"   Relations:       {len(custom_relations)}")
    print(f"   Source chunks:   {len(chunk_payload_map)}")
    print(f"   Relation index:  {len(relation_index)}")

    return (
        custom_entities,
        entity_id_to_node,
        custom_relations,
        relation_metadata,
        entity_descriptions,
        entity_sources,
        relation_index,
        chunk_citation_map,
        chunk_payload_map,
    )
