# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Clinical GraphRAG pipeline for mental-health sources from NIMH, WHO, and DSM. Combines structured graph extraction with community-summarized retrieval for clinical question answering.

## Commands

### Setup
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env  # then edit .env with OPENAI_API_KEY, NEO4J_* credentials
```

### Build the clinical graph (main pipeline)
```bash
python pipeline.py --input-dir .\data\clinical_markdown --source-manifest-path .\data\clinical_source_manifest.json --node-checkpoint-dir .\checkpoints\clinical_pipeline --graph-checkpoint-dir .\checkpoints\clinical_graph --clean-graph-checkpoint-dir .\checkpoints\clinical_graph_clean --community-checkpoint-dir .\checkpoints\clinical_graph_clean --repair-clinical-gaps --clinical-repair-supplemental-node-dir .\checkpoints\pipeline --progress-every 10 --smoke-test
```

### Sync to Neo4j
```bash
python pipeline.py --graph-checkpoint-dir .\checkpoints\clinical_dsm_merged --community-checkpoint-dir .\checkpoints\clinical_dsm_merged --write-neo4j
```

### Run the API
```bash
uvicorn fastapi_app:app --reload
# Docs at http://127.0.0.1:8000/docs
```

The API is Neo4j-only at runtime. Sync the merged graph first, then start the API:
```bash
python pipeline.py --graph-checkpoint-dir .\checkpoints\clinical_dsm_merged --community-checkpoint-dir .\checkpoints\clinical_dsm_merged --write-neo4j
uvicorn fastapi_app:app --reload
```

### Benchmark (proxy mode - no LLM required for evaluation)
```bash
python benchmark_ragas.py --proxy-only --node-checkpoint-dir .\checkpoints\clinical_pipeline --graph-checkpoint-dir .\checkpoints\clinical_graph_clean --community-checkpoint-dir .\checkpoints\clinical_graph_clean --output-dir .\checkpoints\benchmark_clinical_clean_proxy
```

### Repair clinical graph
```bash
python repair_clinical_graph.py
```

### Merge DSM and clinical graphs
```bash
python merge_graph_checkpoints.py --primary-checkpoint-dir .\checkpoints\clinical_graph_clean --secondary-checkpoint-dir .\checkpoints\dsm_graph_v6_repaired --target-checkpoint-dir .\checkpoints\clinical_dsm_merged
```

### Regenerate markdown sources from PDFs/HTML
```bash
python ingest_clinical_sources.py --force
```

### Run tests
```bash
python -m unittest test_clinical_graphrag.py test_pipeline_end_to_end.py -v
```

## Architecture

### Pipeline Stages
1. **Markdown nodes** (`pipeline_helpers.py`): Parse clinical_markdown files into structural nodes
2. **Enrichment** (`pipeline_helpers.py`): Add context via LLM
3. **Graph extraction** (`graphrag_pipeline.py`): Extract entities, relations, citations into pickled artifacts
4. **Cleaning** (`graphrag_pipeline.py`): Remove low-signal nodes/relations
5. **Clinical repair** (`repair_clinical_graph.py`): Targeted fixes for prevalence, differential, autism-treatment gaps
6. **Communities** (`graphrag_pipeline.py`): Leiden partitioning + community summaries
7. **Query** (`graphrag_pipeline.py`): `hybrid_query()` combines community reports + vector retrieval

### Checkpoint System
Pipeline outputs are saved to checkpoint directories as pickled files (`custom_entities.pkl`, `relation_metadata.pkl`, etc.). Checkpoints are identified by existence of required files and can be resumed with `--force-rebuild-*` flags to override.

### GraphRAG Query Flow (`hybrid_query`)
- Classifies query type (symptom/treatment/condition/etc.) via `classify_query`
- Routes to community report retrieval or direct entity lookup
- Falls back to vector similarity search via `retrieval_stack`
- Returns answer, citations, source_chunks, crisis assessment

### Allowed Entity Types
`CONDITION`, `SYMPTOM`, `TREATMENT`, `MEDICATION`, `SAFETY_RESOURCE`, `DEMOGRAPHIC`, `DIAGNOSTIC_CRITERION`, `SPECIFIER`, `RISK_FACTOR`, `COURSE_FEATURE`, `PREVALENCE_STATEMENT`

### Allowed Relations
`HAS_SYMPTOM`, `TREATED_BY`, `PRESCRIBES`, `SUITABLE_FOR`, `CONTRAINDICATED_FOR`, `URGENT_ACTION`, `HAS_DIAGNOSTIC_CRITERION`, `HAS_SPECIFIER`, `HAS_RISK_FACTOR`, `DIFFERENTIAL_DIAGNOSIS`, `COMORBID_WITH`, `HAS_COURSE`, `HAS_PREVALENCE`

## Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | End-to-end CLI orchestrator (stages 1-6) |
| `graphrag_pipeline.py` | Graph extraction, query, communities, cleaning, repair |
| `pipeline_helpers.py` | Markdown parsing, enrichment, node loading |
| `fastapi_app.py` | FastAPI service wrapping hybrid_query |
| `retrieval_stack.py` | Vector retrieval, cross-encoder reranking |
| `benchmark_ragas.py` | RAGAS benchmark runner |
| `repair_clinical_graph.py` | Targeted clinical gap repair |
| `merge_graph_checkpoints.py` | Merge two graph checkpoints |
| `neo4j_helpers.py` | Neo4j sync utilities |
| `safety_shield.py` | Crisis detection and US crisis resources |

## Data

- `data/clinical_markdown/` — ingested NIMH/WHO markdown sources
- `data/source_registry.json` — official NIMH/WHO source registry
- `data/clinical_source_manifest.json` — provenance manifest
- Checkpoints are generated locally and may not be committed

## Environment Variables

- `OPENAI_API_KEY` — required for graph building/rebuilding
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` — for Neo4j sync and Neo4j-backed API queries
- `GRAPH_CHECKPOINT_DIR` or `GRAPH_SYNC_SOURCE_DIR` — optional default checkpoint to sync into Neo4j
