"""Neo4j connection + query helpers for the notebook pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)


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
        with driver.session(database=_neo4j_database()) as session:
            result = session.run("RETURN 'connected' AS status")
            print(f"Neo4j connection: {result.single()['status']}")
            print(f"URI: {_neo4j_uri()}")
            print(f"Database: {_neo4j_database()}")
            return True
    except Exception as exc:
        print(f"Neo4j connection failed: {exc}")
        return False


def write_graph_to_neo4j(
    driver,
    custom_entities,
    seen_relations,
    relation_metadata,
    entity_sources,
    chunk_citation_map=None,
):
    """Write the extracted graph to Neo4j using MERGE semantics."""
    if chunk_citation_map is None:
        chunk_citation_map = {}

    def _write(tx, entities, relations, rel_meta, ent_sources, citation_map):
        for canonical_name, entity in entities.items():
            sources = list(ent_sources.get(canonical_name, set()))
            citations = [citation_map.get(source, "") for source in sources]
            tx.run(
                """
                MERGE (n:Entity {name: $name})
                SET n.type = $type,
                    n.description = $description,
                    n.sources = $sources,
                    n.citations = $citations
                """,
                name=canonical_name,
                type=entity.label,
                description=entity.properties.get("description", ""),
                sources=sources,
                citations=citations,
            )

        for relation_key in relations:
            if not isinstance(relation_key, tuple) or len(relation_key) != 3:
                continue
            subject, relation_label, target = relation_key
            metadata = rel_meta.get((subject, relation_label, target), {})
            query = f"""
                MATCH (a:Entity {{name: $subject}})
                MATCH (b:Entity {{name: $target}})
                MERGE (a)-[r:{relation_label}]->(b)
                SET r.strength = $strength,
                    r.description = $description,
                    r.sources = $sources
            """
            tx.run(
                query,
                subject=subject,
                target=target,
                strength=metadata.get("strength", 5),
                description=metadata.get("description", ""),
                sources=list(metadata.get("sources", set())),
            )

    with driver.session(database=_neo4j_database()) as session:
        session.execute_write(
            _write,
            custom_entities,
            seen_relations,
            relation_metadata,
            entity_sources,
            chunk_citation_map,
        )


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
            """
            CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
            FOR (n:Entity) ON EACH [n.name, n.description]
            """
        )
        session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)")


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
    try:
        query_string = " OR ".join(keywords)
        with driver.session(database=_neo4j_database()) as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('entity_search', $q)
                YIELD node, score
                RETURN node.name AS name, node.type AS type,
                       node.description AS description, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                q=query_string,
                limit=limit,
            )
            return result.data()
    except Exception:
        # Fallback to simple CONTAINS search
        with driver.session(database=_neo4j_database()) as session:
            keyword_conditions = " OR ".join(
                [f"toLower(n.name) CONTAINS toLower('{kw}')" for kw in keywords]
            )
            result = session.run(
                f"""
                MATCH (n:Entity)
                WHERE {keyword_conditions}
                RETURN n.name AS name, n.type AS type,
                       n.description AS description, 1.0 AS score
                LIMIT $limit
                """,
                limit=limit,
            )
            return result.data()


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


# ============================================================
# RESTORE FROM NEO4J
# Run this instead of loading pickles - rebuilds in-memory structures
# ============================================================

def restore_from_neo4j(driver):
    """
    Rebuild custom_entities, entity_id_to_node, custom_relations, 
    and relation_metadata from Neo4j.
    """
    from llama_index.core.graph_stores.types import EntityNode, Relation
    
    custom_entities = {}
    entity_id_to_node = {}
    entity_descriptions = {}
    entity_sources = {}
    custom_relations = []
    relation_metadata = {}
    
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
                   n.description AS desc, n.sources AS sources
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
        
        # Load edges
        edges = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            RETURN a.name AS subj, type(r) AS rel, b.name AS obj,
                   r.strength AS strength, r.description AS desc,
                   r.sources AS sources
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
    
    print(f"✅ Restored from Neo4j:")
    print(f"   Database:  {database}")
    print(f"   Entities:  {len(custom_entities)}")
    print(f"   Relations: {len(custom_relations)}")
    
    return (custom_entities, entity_id_to_node, custom_relations, 
            relation_metadata, entity_descriptions, entity_sources)



