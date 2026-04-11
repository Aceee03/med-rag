# Work Progress

Last updated: 2026-04-11

This file is the first status file to check before starting new work on this project.

## Current System State

- Active default graph: `checkpoints/clinical_dsm_merged`
- API: `fastapi_app.py`
- Main query pipeline: `graphrag_pipeline.py`
- Neo4j exists, but it is not yet the main runtime backend
- Current query behavior supports:
  - graph-backed answers
  - exact supporting `source_chunks`
  - typo-tolerant entity matching
  - natural human symptom questions
  - follow-up symptom refinement with `session_id`

## Current Requested Direction

These are the next required implementation goals.

1. Make Neo4j the main runtime backend for live querying
2. Add a real persistent vector store for raw chunk embeddings using `pgvector` or `Pinecone`
3. Add proper BM25 retrieval as a real retrieval layer
4. Add the LLM safety classifier for ambiguous crisis cases
5. Make the live API use the full hybrid flow:
   - run graph retrieval and vector retrieval in parallel
   - merge results
   - dedupe results
   - rerank results
   - trim to a context budget
   - keep the current citation style

Out of scope for now:

- Phase 6 post-processing work
- Phase 7 persistent conversation management work

## Done

- Ingested clinical sources into markdown from WHO and NIMH
- Built DSM markdown source pipeline
- Attached source metadata to chunks and citations
- Built LLM-based entity and relation extraction
- Used a controlled mental-health relation vocabulary
- Built graph checkpoints locally
- Added graph cleaning
- Added targeted graph repair
- Merged the clinical graph and DSM graph into one checkpoint
- Made the API default to the merged checkpoint when it exists
- Added crisis keyword safety gate with hardcoded crisis response
- Added rule-based query classification
- Added natural human symptom question handling
- Added typo-tolerant entity matching for small misspellings
- Added exact supporting `source_chunks` in `/query`
- Added optional LLM answer synthesis
- Added basic multi-turn follow-up through `session_id`
- Added regression tests for API, typo handling, symptom queries, and session follow-up
- Added Neo4j sync utilities and read/write helper functions

## Partly Done

- Provenance tracking:
  - We keep source labels, file paths, section/header info, chunk text, and chunk ids
  - Page number is not fully carried through the live query response in a clean end-to-end way

- Query analysis:
  - We do classify queries and detect lookup/comparison/symptom flows
  - This is still rule-based, not a full structured LLM analysis schema

- Vector retrieval:
  - We have a retriever in `retrieval_stack.py`
  - This is not the main live API retrieval path

- Reranking:
  - We have reranking in `retrieval_stack.py`
  - It is mainly used in benchmarking, not fully in the live API flow

- Generation:
  - We can synthesize answers with an LLM
  - We do not yet enforce strict inline source ids like `[SRC_001]`

- Multi-turn memory:
  - We keep recent questions in memory by `session_id`
  - This is not persistent storage like Redis

- Neo4j integration:
  - We can sync the graph to Neo4j and query it with helper functions
  - Neo4j is not yet the main live runtime backend

## Not Done Yet

- Neo4j as the main runtime backend for `/query`
- A real persistent vector store such as `pgvector` or `Pinecone`
- Live graph + vector hybrid retrieval in the main API path
- Proper BM25 as a named retrieval component in the main system
- Full merge/dedup/rerank pipeline inside live `/query`
- LLM safety classifier for ambiguous crisis cases
- Fully polished differential ranking for symptom-based diagnosis-style answers

## Deferred For Now

- Inline citation format like `[SRC_001]`
- Citation registry and footnote resolver post-processing
- Output safety scan after generation
- Persistent session store such as Redis
- Broader phase 6 post-processing work
- Broader phase 7 conversation-management work

## Most Recent Completed Work

- Returned exact source chunks instead of only citation labels
- Switched the API default to the merged DSM + clinical graph
- Made natural symptom phrasing count as in-scope
- Added typo tolerance for queries like `depresion`
- Added follow-up symptom context with `session_id`

## Immediate To-Do

- Make Neo4j the main runtime backend
- Add a persistent vector store for chunk embeddings
- Add BM25 retrieval
- Add the LLM safety classifier
- Wire the live API to run graph + vector retrieval in parallel, then merge, dedupe, rerank, and trim

## Notes For Next Work Session

- Keep the current citation style
- Do not work on phase 6 or phase 7 yet
- Implement one concept at a time and explain briefly what changed and where after each concept is finished
