# Clinical GraphRAG

Clinical GraphRAG pipeline for mental-health sources from NIMH, WHO, and DSM-derived material.

PDF ingestion uses `pymupdf4llm`. HTML ingestion uses `trafilatura`. No paid cloud parser is required for source ingestion.

## Requirements

- Python 3.11
- Neo4j, if you want graph sync or browser inspection
- OpenAI API key, if you want to build or rebuild the graph

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

- `OPENAI_API_KEY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## Source Data

The repo includes:

- `data/source_registry.json`: official NIMH/WHO source registry
- `data/clinical_markdown`: ingested markdown sources
- `data/clinical_source_manifest.json`: provenance manifest

If you want to regenerate the markdown sources:

```powershell
python ingest_clinical_sources.py --force
```

## Build The Clinical Graph

This is the main build command:

```powershell
python pipeline.py --input-dir .\data\clinical_markdown --source-manifest-path .\data\clinical_source_manifest.json --node-checkpoint-dir .\checkpoints\clinical_pipeline --graph-checkpoint-dir .\checkpoints\clinical_graph --clean-graph-checkpoint-dir .\checkpoints\clinical_graph_clean --community-checkpoint-dir .\checkpoints\clinical_graph_clean --progress-every 10 --smoke-test
```

That will:

- parse markdown into structural nodes
- enrich nodes with context
- extract graph triplets
- clean the graph
- build community summaries
- run structural smoke tests

## Sync To Neo4j

```powershell
python pipeline.py --graph-checkpoint-dir .\checkpoints\clinical_graph --clean-graph-checkpoint-dir .\checkpoints\clinical_graph_clean --community-checkpoint-dir .\checkpoints\clinical_graph_clean --write-neo4j
```

## Run The API

The API defaults to the cleaned clinical graph checkpoint:

```powershell
uvicorn fastapi_app:app --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

Useful routes:

- `GET /health`
- `GET /meta`
- `GET /safety-resources`
- `POST /query`

Example request body:

```json
{
  "question": "What are the symptoms of PTSD?",
  "answer_with_llm": false
}
```

## Benchmark

Fast proxy benchmark:

```powershell
python benchmark_ragas.py --proxy-only --node-checkpoint-dir .\checkpoints\clinical_pipeline --graph-checkpoint-dir .\checkpoints\clinical_graph_clean --community-checkpoint-dir .\checkpoints\clinical_graph_clean --output-dir .\checkpoints\benchmark_clinical_clean_proxy
```

Outputs:

- `checkpoints/benchmark_clinical_clean_proxy/vector_baseline.jsonl`
- `checkpoints/benchmark_clinical_clean_proxy/vector_reranked.jsonl`
- `checkpoints/benchmark_clinical_clean_proxy/graph_rag.jsonl`
- `checkpoints/benchmark_clinical_clean_proxy/ragas_summary.md`

## Main Files

- `pipeline.py`: end-to-end CLI runner
- `graphrag_pipeline.py`: graph extraction, querying, cleanup, repair, communities
- `pipeline_helpers.py`: markdown parsing and enrichment
- `ingest_clinical_sources.py`: source ingestion
- `fastapi_app.py`: API
- `benchmark_ragas.py`: benchmark runner
- `neo4j_helpers.py`: Neo4j utilities
- `safety_shield.py`: crisis short-circuit logic

## Notes

- Checkpoint folders are generated locally and may not be committed. Build them before running the API if they are missing.
- `benchmark_ragas.py --proxy-only` is the reliable evaluation mode for quick reproduction.
- `CLINICAL_GRAPH_AUDIT.md` contains the latest manual quality summary of the cleaned clinical graph.
