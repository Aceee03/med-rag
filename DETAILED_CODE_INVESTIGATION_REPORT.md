# Clinical GraphRAG Detailed Code Investigation Report

Generated on: 2026-04-23

## 1. Scope and Goal

This report was written after reading and cross-checking the current implementation to support thesis/paper writing.

Primary goal:
- Document what each important file does in practice.
- Capture end-to-end runtime and build behavior.
- Clarify what is actually live vs optional/experimental.
- Provide a bank of likely paper-defense questions with evidence paths.

Files investigated in detail:
- `fastapi_app.py`
- `graphrag_pipeline.py`
- `safety_shield.py`
- `llm_utils.py`
- `pipeline.py`
- `pipeline_helpers.py`
- `ingest_clinical_sources.py`
- `repair_clinical_graph.py`
- `merge_graph_checkpoints.py`
- `neo4j_helpers.py`
- `retrieval_stack.py`
- `benchmark_ragas.py`
- `test_pipeline_end_to_end.py`
- `test_clinical_graphrag.py`

Also investigated because they affect data quality and reproducibility:
- `convert_pdfs_to_markdown.py`
- `dsm_markdown_filter.py`
- `progress_utils.py`
- `README.md`
- `requirements.txt`
- `CURRENT_SYSTEM_OVERVIEW.md`

## 2. System Reality (As Implemented)

The current live query runtime is:
- FastAPI service in `fastapi_app.py`.
- Neo4j-backed graph restoration at startup via `restore_from_neo4j(...)`.
- Query logic in `graphrag_pipeline.py` via `hybrid_query(...)`.
- Rule-based crisis gate in `safety_shield.py`.

Important reality checks from code:
- Neo4j is the required live query backend in production (`runtime_mode: neo4j_only`).
- API startup attempts Neo4j initialization with retries and fails hard if Neo4j is unavailable or graph data is empty.
- Query execution uses `Neo4jGraphQueryBackend` through `graph_backend` injection into `hybrid_query(...)`.
- Checkpoint-backed query mode still exists for offline testing (patched service in end-to-end tests), but not as default production runtime.
- `retrieval_stack.py` exists and is used in benchmarking workflows, not in `hybrid_query(...)` runtime.
- Runtime retrieval is graph-first (Neo4j traversal + community context + local text entity search), not vector-first.

## 3. High-Level Architecture

### 3.1 Runtime Path

1. API request reaches `POST /query` in `fastapi_app.py`.
2. `GraphRAGService.__init__(...)` initializes Neo4j (with retry/backoff), restores graph state from Neo4j, and creates `Neo4jGraphQueryBackend`.
3. `GraphRAGService.query(...)` sends question (plus optional session history) to `hybrid_query(..., graph_backend=self.graph_backend)`.
4. `hybrid_query(...)` calls `classify_query(...)`, which itself uses `assess_safety_risk(...)` first.
5. If crisis: immediate crisis response with safety resources.
6. If out-of-scope: domain boundary response.
7. If in-scope: routing to comparison, reverse symptom, condition forward lookup, or entity lookup.
8. Graph operations dispatch through backend-aware wrappers (Neo4j-backed in production; in-memory fallback only when `graph_backend=None`).
9. Build answer, citations, and exact `source_chunks`.
10. Optional answer synthesis by answer LLM if `answer_with_llm=True` and an LLM client exists.

### 3.2 Build Path

1. Source ingestion (`ingest_clinical_sources.py`) creates markdown corpus + manifest.
2. Node parsing and source metadata attachment (`pipeline_helpers.py`).
3. Node context enrichment (`pipeline_helpers.py`) with resumable checkpointing.
4. Graph extraction (`extract_graph_from_nodes` in `graphrag_pipeline.py`) with schema constraints.
5. Graph cleaning (`clean_graph_artifacts` / `clean_graph_checkpoint`).
6. Optional repair (`repair_graph_checkpoint`) and clinical repair (`repair_clinical_graph.py`).
7. Community detection and summaries (`load_or_build_communities`, `load_or_build_community_summaries`).
8. Optional Neo4j sync (`pipeline.py` + `neo4j_helpers.py`).

### 3.3 Runtime Backend Selection

The runtime query layer supports two modes through `graph_backend`:
- `graph_backend=Neo4jGraphQueryBackend(driver)`:
  - production mode used by API service,
  - forward/entity/reverse/text lookups executed against Neo4j.
- `graph_backend=None`:
  - fallback path used in tests/offline scenarios,
  - lookups executed against in-memory `GraphArtifacts` structures.

This keeps one query orchestration function (`hybrid_query`) while allowing different graph providers.

## 4. Core Data Contracts

### 4.1 API Request/Response

`QueryRequest` fields:
- `question: str`
- `session_id: str | None`
- `answer_with_llm: bool`

`QueryResponse` fields (actual runtime payload contract):
- `source_chunks: list[dict[str, str]]`
  - Each item includes `chunk_id`, `citation`, `text`, `section_title`, `context_tag`, `source_label`, `file_path`, `header_path`.
- `answer: str`
- `query_type: str`
- `citations: list[str]`
- `is_crisis: bool`
- `safety_resources: list[dict[str, str]]`
- `matched_safety_indicators: list[str]`
- `out_of_scope: bool`
- `communities_used: list[str]`

### 4.2 GraphArtifacts (Core In-Memory Graph State)

`GraphArtifacts` in `graphrag_pipeline.py` contains:
- `custom_entities`
- `entity_id_to_node`
- `custom_relations`
- `relation_metadata`
- `entity_descriptions`
- `entity_sources`
- `relation_index`
- `chunk_citation_map`
- `chunk_payload_map`

This structure is the center of both query and graph maintenance operations.

### 4.3 Graph Checkpoint Files

Required files:
- `custom_entities.pkl`
- `entity_id_to_node.pkl`
- `entity_descriptions.pkl`
- `entity_sources.pkl`
- `custom_relations.pkl`
- `relation_metadata.pkl`
- `relation_index.pkl`
- `chunk_citation_map.pkl`

Optional:
- `chunk_payload_map.pkl`

Metadata file:
- `graph_meta.json`

## 5. File-by-File Technical Investigation

### 5.1 `fastapi_app.py`

Role:
- Production HTTP entrypoint around a Neo4j-backed graph query service.

Key implementation points:
- Uses `@lru_cache(maxsize=1)` for singleton `GraphRAGService`.
- Resolves `graph_sync_source_dir` from env/checkpoint candidates for sync/reference metadata.
- Initializes Neo4j on startup with retry/backoff and fails startup if connection/check/data restore fails.
- Restores entities/relations/chunk payloads from Neo4j (`restore_from_neo4j`).
- Sets `query_backend_name = neo4j_live` and `graph_meta.runtime_mode = neo4j_only`.
- Maintains in-memory `session_questions` for follow-up context.
- Truncates session history to last 6 user questions.

Exposed routes:
- `GET /`
- `GET /favicon.ico`
- `GET /health`
- `GET /meta`
- `GET /safety-resources`
- `POST /query`

Paper-relevant notes:
- Stateless API semantics at HTTP level, with process-local session memory for follow-up interpretation.
- This design improves conversational continuity but is not durable across restarts and has no TTL.

Potential limitations:
- No per-session eviction policy.
- No explicit API rate limiting in code.
- Runtime now depends on Neo4j availability and synced graph data.

### 5.2 `graphrag_pipeline.py`

Role:
- Core graph extraction, cleaning, merge, repair, community, and query logic.

Schema and constraints:
- `ALLOWED_TYPES`: 11 entity classes including DSM/clinical-specific abstractions.
- `ALLOWED_RELATIONS`: 13 relation labels.
- `VALID_TRIPLET_TYPES`: explicit allowed `(subject_type, relation, object_type)` combinations.

Normalization and quality mechanisms:
- Canonicalization (`normalize_lookup_text`, `canonicalize`).
- Abbreviation expansion (`ABBREVIATION_MAP`).
- Garbage filtering (`is_garbage_entity`).
- Placeholder and low-signal filtering (`is_placeholder_entity_name`, `is_low_signal_entity_name`).
- Relation strictness check in post-cleanup (`_is_strictly_valid_post_cleanup_relation`).

Extraction path:
- `extract_graph_from_nodes(...)` uses LLM prompt (`EXTRACTION_PROMPT`) per node.
- Parses JSON payload with fallback extraction if fenced/noisy.
- Tracks quality stats with `GraphExtractionStats`.
- Supports resumable extraction via partial checkpoint files.

Repair path:
- `repair_graph_artifacts(...)` applies targeted repair prompts for selected conditions/relations.
- Uses relevance scoring of enriched nodes (`select_repair_nodes`) before repair LLM call.
- Applies extracted payload via `_apply_extracted_payload_to_state(...)`.

Cleaning and merge path:
- `merge_equivalent_entities(...)` merges aliases by compact canonical form.
- `clean_graph_artifacts(...)` removes low-value entities/relations and isolated nodes.
- `merge_graph_artifacts(...)` merges two graphs with conflict accounting and post-merge cleanup.

Community path:
- Detects communities with Leiden (`detect_communities`).
- Summarizes communities with LLM when provided, heuristic fallback otherwise.

Query path (`hybrid_query(...)`):
- Steps:
  1) classify (crisis / out_of_scope / in_scope)
  2) entity matching (exact + alias + fuzzy)
  3) route to comparison, reverse symptom, forward condition, or entity lookup
  4) add optional global community context
  5) optional LLM synthesis over assembled context
- Produces citation strings and source chunk payloads from chunk maps.
- Backend wrappers (`_forward_lookup_with_backend`, `_entity_relation_lookup_with_backend`, `_reverse_symptom_lookup_with_backend`, `_text_search_entities_with_backend`) route graph operations to Neo4j when `graph_backend` is provided.

Critical reality:
- No direct vector retrieval call in `hybrid_query(...)`.
- Runtime uses graph + lightweight text scoring over graph entities + community summaries.
- In production API mode, graph operations are Neo4j-backed via injected backend.

Potential limitations:
- Heavy reliance on rule-based query intent + fuzzy matching thresholds.
- LLM extraction quality and prompt adherence remain a key dependency.

### 5.3 `safety_shield.py`

Role:
- Crisis risk short-circuit before normal answer generation.

How it works:
- Regex sets for:
  - informational phrasing,
  - first-person indicators,
  - crisis signals (`explicit_self_harm_intent`, `imminence_or_plan`, `severe_distress`).
- Returns:
  - `is_crisis`
  - `matched_indicators`
  - `resources`
  - `response`

Emergency resources are hardcoded in `US_CRISIS_RESOURCES`.

Potential limitations:
- Rule-based patterns can miss nuanced language.
- No multilingual logic in this file.

### 5.4 `llm_utils.py`

Role:
- Shared OpenAI client builder with prefixed env configuration.

Behavior:
- Returns `None` if `OPENAI_API_KEY` missing.
- Reads `{PREFIX}_MODEL`, `{PREFIX}_MAX_TOKENS`, optional `{PREFIX}_REASONING_EFFORT`.
- Uses graceful fallback if the installed OpenAI wrapper does not accept `reasoning_effort`.

Paper-relevant point:
- Centralized configuration allows stage-specific models (enrich, graph extract, repair, answer).

### 5.5 `pipeline.py`

Role:
- CLI orchestrator for end-to-end build, cleanup, repair, summaries, query smoke tests, and optional Neo4j sync.

Main orchestrated stages:
- Stage 1: markdown nodes
- Stage 2: enrichment
- Stage 3: graph extraction
- Stage 3B: clean graph
- Stage 3C: repair DSM gaps (optional)
- Stage 3D: repair clinical gaps (optional)
- Stage 4: communities + summaries
- Stage 5: Neo4j sync (optional)
- Stage 6: one-off query or smoke tests (optional)

Engineering characteristics:
- Resume/rebuild flags per stage.
- Explicit guardrails when an LLM is required but unavailable.
- Line-buffered stdout for easier progress logs.

### 5.6 `pipeline_helpers.py`

Role:
- Node-level preprocessing, source metadata attachment, enrichment checkpointing.

Key mechanisms:
- Source state fingerprinting to validate cache reuse.
- Source manifest mapping by resolved path and filename.
- Citation label construction from authority/title/header context.
- Enrichment prompting with fallback context when LLM fails.
- Resumable enrichment with metadata compatibility checks (`nodes_hash`, prompt version, model name).

Paper-relevant point:
- Provenance and source labeling are integrated early (before graph extraction), improving downstream citation traceability.

### 5.7 `ingest_clinical_sources.py`

Role:
- Pull clinical sources from registry and convert to markdown corpus + manifest.

Behavior:
- Supports source filters by authority/condition and optional limit.
- HTML path: `trafilatura` extraction.
- PDF path: conversion through `convert_pdfs_to_markdown` helpers.
- Writes per-source output markdown and aggregate `clinical_source_manifest.json`.

Paper-relevant point:
- Reproducible source acquisition pipeline with stored parser/provenance fields.

### 5.8 `repair_clinical_graph.py`

Role:
- Clinical repair wrapper combining model-based repair and curated fact injection.

What is distinctive here:
- Hardcoded `CURATED_RELATION_RESETS` clears selected relation families before curation.
- `CURATED_PAYLOADS` injects high-priority facts for:
  - depression prevalence,
  - GAD prevalence + differential diagnosis,
  - PTSD differential diagnosis,
  - autism treatment coverage.
- Loads repair nodes from clinical and optional supplemental node checkpoints, then deduplicates.

Pipeline inside this script:
1) load source graph,
2) run `repair_graph_artifacts(...)` for `CLINICAL_REPAIR_TARGETS`,
3) apply curated payloads,
4) clean graph,
5) refresh citations,
6) save with repair lineage metadata.

Paper-relevant point:
- Hybrid strategy: LLM repair + deterministic curated corrections for known critical factual gaps.

### 5.9 `merge_graph_checkpoints.py`

Role:
- CLI tool to merge two checkpoints into one cleaned final graph.

Behavior:
- Delegates to `merge_graph_checkpoints(...)` in `graphrag_pipeline.py`.
- Prints merge report: entity overlaps, type conflicts, relation merge stats, cleanup summary.

### 5.10 `neo4j_helpers.py`

Role:
- Neo4j write/read/query helpers for runtime queries and graph synchronization.

Capabilities:
- Driver creation + health test.
- Full graph write using MERGE semantics.
- Clearing graph and creating indexes/constraints.
- Query helpers: forward, reverse, comparison, full-text, neighborhood, full restore to in-memory structures.
- `Neo4jGraphQueryBackend` class for production query-time traversal.

`Neo4jGraphQueryBackend` methods:
- `forward_lookup(...)`
- `entity_relation_lookup(...)`
- `reverse_symptom_lookup(...)`
- `text_search_entities(...)`

This class is injected into `hybrid_query(...)` by API service and is the active production graph query provider.

Current system position:
- Required for production runtime query path.
- Also used for sync, inspection, and restoration workflows.

Potential risks worth noting:
- Runtime dependency on Neo4j health and data availability.
- Full-text fallback behavior should still be reviewed for query performance under load.

### 5.11 `retrieval_stack.py`

Role:
- Vector and reranking module for retrieval experiments and benchmark usage.

Components:
- `VectorRetriever`
  - primary backend: sentence-transformers embeddings,
  - fallback backend: lexical TF-IDF-like scoring.
- `CrossEncoderReranker`
  - cross-encoder model if available,
  - fallback overlap-based reranking.
- Context rendering + answer synthesis helper.

Current system position:
- This stack is active in benchmarking and can be used for experiments.
- Not integrated directly into `hybrid_query(...)` runtime currently.

### 5.12 `benchmark_ragas.py`

Role:
- Comparative benchmark runner (vector baseline, reranked vector, graph pipeline).

Notable mechanics:
- Enforces node-hash compatibility between node checkpoint and graph checkpoint.
- Saves per-system prediction JSONL outputs.
- Supports full RAGAS metrics when dependencies/config available.
- Falls back to deterministic proxy token-overlap metrics otherwise.

Paper-relevant point:
- Provides reproducible evaluation harness even without external evaluator availability (`--proxy-only`).

### 5.13 `test_pipeline_end_to_end.py`

Role:
- Integration/regression suite covering API and pipeline orchestration behavior.

Covered behaviors:
- API `/health` and `/meta` sanity checks.
- Query route behavior for prevalence, human symptom phrasing, typo tolerance, session follow-up.
- One-command clinical repair rebuild through `pipeline.py`.
- Neo4j sync orchestration via mocks.
- Graph merge conflict handling behavior.

### 5.14 `test_clinical_graphrag.py`

Role:
- Focused regression checks for core GraphRAG answer quality constraints.

Checks include:
- In-scope classification for medication query.
- Citation normalization.
- Curated prevalence facts.
- Differential diagnosis terms.
- Autism treatment output excludes screening/evaluation noise.
- Comparison answers omit known noisy terms.

### 5.15 `convert_pdfs_to_markdown.py` (supporting preprocessing)

Role:
- Robust PDF-to-markdown converter with manifest generation.

Notable engineering choices:
- Runtime patch to enforce `int64` edge index dtype for layout ONNX path.
- Automatic fallback from layout mode to legacy extraction mode.
- File collision-safe output naming + manifest with page chunks and metadata.

### 5.16 `dsm_markdown_filter.py` (supporting preprocessing)

Role:
- Filters noisy DSM markdown before graph extraction.

Two-layer filtering:
1) Section filtering (focus on Section II by default, remove administrative content).
2) Chunk quality filtering (length, table/list heaviness, alpha ratio, clinical signal presence).

Output:
- filtered markdown file,
- JSON audit report with discard reasons and chunk-level diagnostics.

### 5.17 `progress_utils.py`

Role:
- Shared progress printer with elapsed/rate/ETA computation used across long pipeline steps.

## 6. End-to-End Query Behavior (Detailed)

Inside `hybrid_query(...)`:

1. Compose `analysis_text` from conversation history + current question.
2. `classify_query(...)` decides `CRISIS`, `OUT_OF_SCOPE`, or `IN_SCOPE`.
3. If crisis:
   - return crisis response/resources,
   - no graph retrieval path executed.
4. If out-of-scope:
   - return domain boundary response,
   - no graph retrieval path executed.
5. In-scope routing uses:
   - comparison intent patterns,
   - symptom phrasing patterns + matched symptoms,
   - relation-intent hints,
   - matched condition/entity categories.
6. Build `multipath_result` according to branch.
7. Optionally enrich answer with:
   - local entity text search rows,
   - community summary matches.
8. Convert chunk IDs to citation strings and source payloads.
9. If LLM synthesis enabled, generate final answer from structured context blocks.

Backend dispatch detail:
- In production API service, `graph_backend` is a Neo4j backend object, so graph operations are served from Neo4j.
- In offline/test mode where `graph_backend=None`, the same orchestration falls back to in-memory graph functions.

`query_type` values observed from code branches:
- `crisis`
- `out_of_scope`
- `comparison`
- `reverse_symptom`
- `forward_lookup` (default when condition recognized)
- relation-intent labels such as `prevalence`, `treatment`, `symptoms`, `diagnostic_criteria`, etc.
- `entity_lookup`
- `general`

## 7. End-to-End Build Behavior (Detailed)

The build process is resumable at multiple points:
- node/doc caching based on source fingerprint,
- enrichment caching based on nodes hash/model/prompt version,
- graph extraction partial-state snapshots for interrupted runs.

Quality-control layers in build path:
- constrained extraction schema,
- garbage/low-signal filtering,
- post-clean validity checks,
- targeted repair passes,
- curated payload injection for known weak areas,
- optional structural smoke tests.

## 8. Safety Behavior (Detailed)

`assess_safety_risk(...)` logic summary:
- detect indicators,
- suppress some false positives for purely informational queries,
- elevate if first-person language and/or imminence/severe-distress cues exist.

If crisis flagged in runtime:
- graph retrieval is bypassed,
- response contains emergency resources and indicator labels.

Design implication for paper:
- This is a practical guardrail layer, not clinical triage software.

## 9. Testing and Evaluation Coverage Map

Implemented coverage:
- API route health/meta/query behavior.
- Session-based follow-up behavior.
- Pipeline repair orchestration.
- Neo4j sync orchestration contract.
- Graph merge conflict behavior.
- Key answer regression expectations on selected clinical queries.
- Checkpoint-backed service patch used in API end-to-end tests to avoid requiring live Neo4j for every test run.

Evaluation harness:
- Benchmark runner compares vector baseline, reranked vector, and graph answers.
- Supports RAGAS when available; deterministic proxy metrics otherwise.

Gaps not deeply covered by current tests:
- Adversarial prompt behavior.
- Latency and throughput benchmarks under load.
- Safety behavior across multilingual/implicit self-harm language.
- Long-session memory growth behavior.

## 10. Risks, Limitations, and Technical Debt

1. Session memory persistence and growth
- Process-local in-memory map; no TTL/eviction strategy beyond per-session list truncation.

2. Rule-heavy query and safety routing
- Robust for known patterns, but less flexible for unexpected phrasing.

3. LLM dependency in extraction/repair quality
- Deterministic prompts still depend on model output quality.

4. Runtime retrieval architecture
- Graph-first runtime currently does not include vector retrieval in the live path.

5. Neo4j operational dependency
- API runtime is Neo4j-only in production mode.
- Startup fails if Neo4j is unavailable or graph restore returns empty graph.
- Operational reliability now depends on database availability and sync freshness.

6. Curated payload maintenance burden
- Handcrafted curated payloads are effective but require versioned maintenance discipline.

## 11. Information Questions for Paper/Defense (With Evidence Paths)

Each question below includes:
- Why it matters.
- Short answer from the implementation.
- Where to cite in code.

### Q1. What is the actual runtime architecture today?
- Why it matters: avoids overclaiming hybrid/vector/DB capabilities.
- Short answer: FastAPI + Neo4j-backed graph restore/query backend + rule-based safety + graph query routing.
- Evidence: `fastapi_app.py`, `graphrag_pipeline.py`, `CURRENT_SYSTEM_OVERVIEW.md`.

### Q2. Is Neo4j required for serving `/query`?
- Why it matters: deployment and reproducibility.
- Short answer: Yes for production API runtime in current code. The service initializes Neo4j at startup, retries on transient failure, and fails hard if graph data is unavailable.
- Evidence: `fastapi_app.py` (`_init_neo4j`), `neo4j_helpers.py`, `README.md`.

### Q3. Is vector retrieval used in live API queries?
- Why it matters: claims about retrieval method.
- Short answer: Not directly in `hybrid_query(...)`; vector stack is separate.
- Evidence: `graphrag_pipeline.py` (no retrieval_stack usage), `retrieval_stack.py`, `benchmark_ragas.py`.

### Q4. How are crisis queries handled?
- Why it matters: safety section in paper.
- Short answer: regex-based crisis detection; immediate short-circuit response with resources.
- Evidence: `safety_shield.py`, `graphrag_pipeline.py` (`hybrid_query` early return on crisis).

### Q5. How does the system classify in-scope vs out-of-scope questions?
- Why it matters: domain boundary and failure modes.
- Short answer: keywords + natural symptom heuristics + relation-intent detection + entity matching.
- Evidence: `graphrag_pipeline.py` (`classify_query`, `_looks_like_natural_symptom_question`, `detect_relation_intent`, `match_entities_in_text`).

### Q6. What enforces graph schema quality?
- Why it matters: trustworthiness.
- Short answer: allowlisted entity types, relations, and valid triplet-type combinations.
- Evidence: `graphrag_pipeline.py` (`ALLOWED_TYPES`, `ALLOWED_RELATIONS`, `VALID_TRIPLET_TYPES`).

### Q7. How does the system reduce noisy entities/edges?
- Why it matters: graph precision.
- Short answer: placeholder/garbage/low-signal filters + strict relation validation + isolated-node pruning.
- Evidence: `graphrag_pipeline.py` (`is_garbage_entity`, `is_placeholder_entity_name`, `is_low_signal_entity_name`, `clean_graph_artifacts`).

### Q8. How are entities normalized across naming variants?
- Why it matters: dedup and retrieval consistency.
- Short answer: canonicalization + abbreviation expansion + alias merge logic.
- Evidence: `graphrag_pipeline.py` (`canonicalize`, `ABBREVIATION_MAP`, `merge_equivalent_entities`).

### Q9. How is provenance preserved for answers?
- Why it matters: explainability and auditability.
- Short answer: chunk-level citation and payload maps are carried through and returned in `/query`.
- Evidence: `pipeline_helpers.py` (citation construction), `graphrag_pipeline.py` (`build_chunk_payload_map`, `source_chunk_payloads`), `fastapi_app.py` response mapping.

### Q10. How are follow-up symptom queries supported?
- Why it matters: conversational utility claim.
- Short answer: session history by `session_id` is prepended to analysis text in query classification/routing.
- Evidence: `fastapi_app.py` (`session_questions`), `graphrag_pipeline.py` (`analysis_text` in `hybrid_query`).

### Q11. How reproducible is the build process?
- Why it matters: thesis reproducibility chapter.
- Short answer: deterministic checkpoints with hashes/version metadata and resume support.
- Evidence: `pipeline_helpers.py` (source fingerprint, nodes hash), `graphrag_pipeline.py` (partial state/meta, checkpoint metadata), `pipeline.py` flags.

### Q12. How are source documents ingested and normalized?
- Why it matters: data pipeline transparency.
- Short answer: WHO/NIMH registry ingestion, HTML via trafilatura, PDF conversion via pymupdf4llm path.
- Evidence: `ingest_clinical_sources.py`, `convert_pdfs_to_markdown.py`, `data/source_registry.json`.

### Q13. How are known factual gaps corrected?
- Why it matters: post-extraction quality strategy.
- Short answer: targeted repair + curated payload injection + cleanup.
- Evidence: `repair_clinical_graph.py`, `graphrag_pipeline.py` repair helpers.

### Q14. Which clinical facts are currently curated by design?
- Why it matters: distinguishes extracted vs hand-curated knowledge.
- Short answer: depression prevalence, GAD prevalence/differential, PTSD differential, autism treatment set.
- Evidence: `repair_clinical_graph.py` (`CURATED_PAYLOADS`).

### Q15. How are two graph checkpoints merged safely?
- Why it matters: DSM + clinical integration validity.
- Short answer: merge entities/relations/sources, keep primary label on conflicts, then run cleanup.
- Evidence: `graphrag_pipeline.py` (`merge_graph_artifacts`, `merge_graph_checkpoints`), `merge_graph_checkpoints.py`.

### Q16. What exactly is tested in integration/regression suites?
- Why it matters: rigor claims.
- Short answer: API routes, query behavior, session follow-up, clinical repair rebuild, Neo4j sync orchestration, merge conflict behavior, curated fact checks.
- Evidence: `test_pipeline_end_to_end.py`, `test_clinical_graphrag.py`.

### Q17. What does benchmark evaluation compare?
- Why it matters: experimental design section.
- Short answer: vector baseline vs reranked vector vs graph pipeline outputs.
- Evidence: `benchmark_ragas.py` (`run_systems`).

### Q18. What if RAGAS is unavailable in environment?
- Why it matters: reproducibility under dependency variance.
- Short answer: fallback to deterministic proxy token-overlap metrics.
- Evidence: `benchmark_ragas.py` (`evaluate_with_ragas`, `_proxy_metric_summary`, `--proxy-only`).

### Q19. What are the most important configurable knobs?
- Why it matters: ablation and setup reporting.
- Short answer: stage-specific model settings, pipeline force/resume flags, checkpoint paths, answer/smoke/neo4j toggles.
- Evidence: `llm_utils.py`, `pipeline.py`, `fastapi_app.py` env-based checkpoint resolution.

### Q20. What are realistic limitations to disclose in the paper?
- Why it matters: credibility.
- Short answer: regex safety constraints, rule-heavy routing, LLM extraction dependency, non-persistent session memory, Neo4j availability dependency, runtime not vector+graph hybrid yet.
- Evidence: `safety_shield.py`, `graphrag_pipeline.py`, `fastapi_app.py`, `retrieval_stack.py` vs runtime path.

### Q21. How are citation labels made readable for humans?
- Why it matters: usability and audit process.
- Short answer: source label + normalized section labels with structured authority naming.
- Evidence: `pipeline_helpers.py` (`build_source_label`, `build_citation_label`, `citation_from_metadata`).

### Q22. How does the system handle extraction interruptions?
- Why it matters: long-running pipeline robustness.
- Short answer: writes partial state and can resume with compatibility checks.
- Evidence: `graphrag_pipeline.py` (`_snapshot_graph_state`, `_load_partial_graph_state`, `extract_graph_from_nodes`).

### Q23. How are symptom aliases handled for natural language queries?
- Why it matters: practical query UX.
- Short answer: phrase alias table maps colloquial symptoms to graph symptom entities.
- Evidence: `graphrag_pipeline.py` (`SYMPTOM_PHRASE_ALIASES`, `_match_symptom_aliases`).

### Q24. How are community summaries generated when no summary LLM is configured?
- Why it matters: fallback behavior.
- Short answer: heuristic summary from member labels and relation patterns.
- Evidence: `graphrag_pipeline.py` (`_heuristic_community_summary`, `load_or_build_community_summaries`).

### Q25. How can you justify that answers are grounded and not free text generation?
- Why it matters: grounding claim.
- Short answer: fallback answers are assembled from graph/entity/community/source context; optional LLM synthesis still receives bounded context and includes citations/source chunks in output payload.
- Evidence: `graphrag_pipeline.py` (`_fallback_answer`, `source_chunk_payloads`, LLM synthesis block in `hybrid_query`), `fastapi_app.py` response contract.

### Q26. How does the system support both production and offline graph query modes?
- Why it matters: clarifies architecture and testing strategy.
- Short answer: `hybrid_query(...)` accepts an optional `graph_backend`; production injects `Neo4jGraphQueryBackend`, while tests/offline runs can pass `None` to use in-memory graph functions.
- Evidence: `fastapi_app.py`, `graphrag_pipeline.py`, `neo4j_helpers.py`, `test_pipeline_end_to_end.py`.

## 12. Suggested Tables/Figures for the Paper (Directly Supported by Code)

1. Architecture figure
- Ingestion -> Node enrichment -> Graph extraction -> Cleaning/Repair -> Query API.

2. Schema table
- Entity types, relation labels, and valid triplet constraints.

3. Runtime routing table
- Crisis / out-of-scope / comparison / reverse symptom / forward / entity lookup.

4. Query backend mode table
- Production API mode: Neo4j backend dispatch.
- Offline/test mode: in-memory fallback dispatch.

5. Checkpoint artifact table
- Required/optional files and what each stores.

6. Evaluation protocol table
- vector baseline, reranked vector, graph pipeline, metric backend behavior.

7. Safety logic table
- Indicator categories and final decision conditions.

## 13. Reproducibility Commands (As Documented in Project)

Source ingestion:
- `python ingest_clinical_sources.py --force`

Main build example:
- `python pipeline.py --input-dir .\data\clinical_markdown --source-manifest-path .\data\clinical_source_manifest.json --node-checkpoint-dir .\checkpoints\clinical_pipeline --graph-checkpoint-dir .\checkpoints\clinical_graph --clean-graph-checkpoint-dir .\checkpoints\clinical_graph_clean --community-checkpoint-dir .\checkpoints\clinical_graph_clean --repair-clinical-gaps --clinical-repair-supplemental-node-dir .\checkpoints\pipeline --progress-every 10 --smoke-test`

Run API:
- `python pipeline.py --graph-checkpoint-dir .\checkpoints\clinical_dsm_merged --community-checkpoint-dir .\checkpoints\clinical_dsm_merged --write-neo4j`
- `uvicorn fastapi_app:app --reload`

Benchmark (proxy mode):
- `python benchmark_ragas.py --proxy-only --node-checkpoint-dir .\checkpoints\clinical_pipeline --graph-checkpoint-dir .\checkpoints\clinical_graph_clean --community-checkpoint-dir .\checkpoints\clinical_graph_clean --output-dir .\checkpoints\benchmark_clinical_clean_proxy`

Tests:
- `python -m unittest test_clinical_graphrag.py test_pipeline_end_to_end.py -v`

## 14. Final Takeaway

This repository already implements a serious, reproducible clinical GraphRAG workflow with:
- structured extraction constraints,
- practical safety gate,
- explicit provenance,
- targeted repair strategy,
- integration tests and benchmark tooling.

The strongest scientific framing for the paper is:
- a graph-grounded clinical QA system with repair-aware quality controls,
- currently served through a Neo4j-backed runtime query backend,
- with vector benchmarking capabilities present as adjacent (non-primary runtime) components.
