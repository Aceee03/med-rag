# Current System Overview

Last updated: 2026-04-11

This file explains the main files in the project and how the system works right now.

## Main Files

### Runtime

- `fastapi_app.py`
  - The API entry point
  - Serves `/query`, `/health`, `/meta`, and `/safety-resources`

- `graphrag_pipeline.py`
  - The main brain of the system
  - Loads graph checkpoints
  - Classifies queries
  - Matches entities and symptoms
  - Runs graph retrieval
  - Collects citations and exact `source_chunks`
  - Optionally asks an LLM to write the final answer

- `safety_shield.py`
  - The current safety gate
  - Uses rules and crisis keywords to stop unsafe queries early
  - Returns crisis resources when needed

- `llm_utils.py`
  - Builds the OpenAI client used in different parts of the project

### Build / Graph Construction

- `pipeline.py`
  - Main build pipeline
  - Turns markdown sources into nodes
  - Enriches them
  - Extracts the graph
  - Cleans and repairs it
  - Can sync the graph to Neo4j

- `pipeline_helpers.py`
  - Helper code for loading markdown, attaching metadata, chunking, and checkpointing

- `ingest_clinical_sources.py`
  - Downloads and converts WHO and NIMH sources into markdown

- `repair_clinical_graph.py`
  - Targeted repair pass for missing or weak clinical facts

- `merge_graph_checkpoints.py`
  - Merges DSM and clinical graph checkpoints into one final graph

### Support / Retrieval / Storage

- `neo4j_helpers.py`
  - Neo4j read/write helpers
  - Useful today, but not the live runtime backend yet

- `retrieval_stack.py`
  - Vector retrieval and reranking code
  - Exists, but is not the main `/query` path yet

- `benchmark_ragas.py`
  - Evaluation and benchmark runner
  - Used for comparing graph answers and vector retrieval behavior

### Tests

- `test_pipeline_end_to_end.py`
  - API and pipeline regression tests

- `test_clinical_graphrag.py`
  - GraphRAG behavior tests

## Main Data And Runtime Directories

- `data/clinical_markdown`
  - WHO and NIMH markdown sources

- `data/markdown_filtered/DSM.filtered.md`
  - Filtered DSM markdown source

- `checkpoints/clinical_dsm_merged`
  - Main merged graph checkpoint
  - This is the default graph the API currently uses

## How The System Works Right Now

The system currently works in this order:

1. Source documents are converted into markdown
2. The markdown is split into nodes or chunks
3. Each chunk gets source metadata
4. An LLM extracts entities and relations from those chunks
5. The graph is cleaned and repaired
6. The final graph is saved as a checkpoint
7. The API loads that checkpoint into memory at startup
8. When a user asks a question, the safety layer runs first
9. If the question is safe, the query pipeline decides what type of graph lookup to run
10. The system returns an answer, citations, and exact supporting source chunks

## Query Flow

When a question reaches the API:

1. `fastapi_app.py` receives the request
2. It calls `hybrid_query(...)` in `graphrag_pipeline.py`
3. `safety_shield.py` checks whether the question looks like a crisis query
4. If the query is safe, the system classifies it and routes it

Current routing behavior:

- One condition mentioned:
  - forward lookup

- Symptoms mentioned:
  - reverse symptom lookup

- Two conditions compared:
  - comparison lookup

- Treatment or medication mentioned:
  - entity lookup

- Not recognized as a mental-health question:
  - out-of-scope response

## What The Answer Contains

After retrieval, the system builds:

- a graph-based result
- `citations`
- `source_chunks`

Then:

- if `answer_with_llm = false`
  - it returns a graph-based fallback answer

- if `answer_with_llm = true`
  - it sends the retrieved context to an LLM
  - the LLM writes a cleaner final answer

If the same `session_id` is reused, the system also looks at recent previous questions so follow-up symptom messages can refine the answer.

## Important Current Reality

Right now, the live runtime backend is not Neo4j yet.

The live API is using the saved graph checkpoint loaded into memory through:

- `fastapi_app.py`
- `graphrag_pipeline.py`

Also right now:

- vector retrieval exists, but it is mostly separate in `retrieval_stack.py`
- reranking exists, but it is mostly used in `benchmark_ragas.py`
- BM25 is not a proper live layer yet
- Neo4j sync exists, but Neo4j is not the live query engine yet

## Simple Summary

The system today is:

- `fastapi_app.py` runs the app
- `graphrag_pipeline.py` answers the questions
- `pipeline.py` builds the graph
- `safety_shield.py` protects the input

The API currently answers from the merged local graph checkpoint, not from Neo4j and not from a full graph + vector hybrid runtime yet.
