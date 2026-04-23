Project Technical Sheet

Last updated: April 18, 2026

General Information

Project title: Evaluating Retrieval-Augmented Generation for Medical Question Answering and Literature Summarization.

Current prototype focus: Clinical GraphRAG for mental-health question answering.

Institution: Higher School of Computer Science and Digital Technologies, ESTIN Bejaia.

Specialty: Artificial Intelligence and Data Sciences.

Prepared by: Boumediene Wadia and Boudebouz Yacine.

Supervisor: Dr. Lekehali Somia.

Academic year: 2025/2026.

Project Summary

This project aims to design, test, and evaluate a GraphRAG system for medical question answering, with a prototype application in the mental-health domain. Its main purpose is to reduce hallucinations produced by large language models by grounding answers in a structured knowledge base built from official sources such as WHO, NIMH, and DSM-derived material.

Instead of relying only on semantically similar text retrieval, the system transforms documents into entities, relations, graph structures, citations, and exact source chunks that can be used at query time. This makes the final answers more explainable, more traceable, and safer.

Problem Statement

Large language models are highly capable, but they can still generate unsupported or hallucinated answers, especially in sensitive domains such as medicine. This project addresses the following question: can a GraphRAG approach improve the accuracy, traceability, and reliability of medical answers compared with a traditional vector-based RAG system?

Project Objectives

The general objective of the project is to evaluate the contribution of a GraphRAG architecture for medical question answering, especially for mental-health questions.

The specific objectives are to build a reliable corpus from official medical sources, convert source documents into machine-usable markdown, segment and enrich the content with structural context, extract entities and relations automatically using an LLM, build and clean a clinical knowledge graph, merge the clinical graph and DSM graph into a final checkpoint, query the graph through a FastAPI service, return citations and exact supporting passages, compare GraphRAG against vector-based baselines, and add safety mechanisms for sensitive or crisis-related queries.

Domain and Data Sources

The prototype focuses on mental health as its main case study. The sources used include official NIMH documents, official WHO documents, and filtered DSM content converted into markdown.

At the current stage, the official clinical corpus contains 10 NIMH/WHO documents. These documents are converted into markdown and then transformed into structured nodes and chunks. The chunks preserve provenance metadata such as source label, file path, section or header information, supporting text, and chunk identifiers.

Technologies and Tools Used

The project is built mainly with Python 3.11 and executed locally through PowerShell, with environment configuration managed through a .env file.

The main libraries used in the project include llama-index-core, llama-index-llms-openai, llama-index-llms-gemini, google-generativeai, python-dotenv, pymupdf4llm, trafilatura, requests, networkx, graspologic, numpy, fastapi, uvicorn, sentence-transformers, and ragas.

For storage and runtime, the system uses Neo4j for graph synchronization and live querying, while local checkpoints are used to store intermediate and final pipeline artifacts.

Overall System Architecture

The system is organized into two main phases.

The first phase is the offline graph construction pipeline. It includes clinical source ingestion, PDF or HTML to markdown conversion, structural chunking, context enrichment, LLM-based entity and relation extraction, graph cleaning, targeted clinical repair, community detection, community summary generation, checkpoint saving, and finally the merge of the clinical and DSM graphs.

The second phase is the online query phase. In this phase, a user question is received through FastAPI, passes through a safety layer, is classified, and then routed to graph retrieval logic. The system returns an answer together with citations and source chunks, and can optionally use an LLM to synthesize the final response.

Main Project Files

The main files of the project are pipeline.py, which handles the end-to-end build and orchestration process; graphrag_pipeline.py, which contains the core GraphRAG logic, extraction, cleanup, and querying; fastapi_app.py, which serves as the API entry point; ingest_clinical_sources.py, which ingests WHO and NIMH sources; pipeline_helpers.py, which provides parsing, chunking, metadata, and checkpoint helpers; repair_clinical_graph.py, which performs targeted graph repair; merge_graph_checkpoints.py, which merges the clinical and DSM graphs; neo4j_helpers.py, which contains Neo4j read and write helpers; retrieval_stack.py, which contains vector retrieval and reranking components; benchmark_ragas.py, which runs evaluation and benchmarking; and safety_shield.py, which implements the safety gate and crisis-response logic.

Work Already Completed

Several major parts of the project have already been completed. On the data and graph construction side, official clinical sources were ingested into markdown, the DSM markdown pipeline was built, provenance metadata was attached to chunks, contextual enrichment was added, LLM-based entity and relation extraction was implemented, a controlled mental-health schema was adopted, local graph checkpoints were built, graph cleaning was added, targeted graph repair was implemented, and the clinical graph was merged with the DSM graph into a final checkpoint.

On the runtime side, the FastAPI service is operational and the runtime is now Neo4j-only. The system supports symptom, condition, treatment, and comparison questions. It includes typo-tolerant entity matching, natural symptom phrasing support, citations, exact source chunks, optional LLM answer synthesis, and basic multi-turn context through session_id.

On the quality and safety side, a crisis-keyword safety layer was added, regression tests for the API and pipeline were written, tests for typo handling and session follow-up were added, graph synchronization to Neo4j was implemented, and Neo4j storage for source chunks was added.

Work Partially Completed

Some parts of the system are only partially complete. Several provenance fields are preserved, but a clean end-to-end flow for fields such as page numbers is not fully finalized. Query analysis exists, but it is still mostly rule-based. Vector retrieval is available in retrieval_stack.py, but it is not yet the main live API path. Reranking is implemented but mainly used in benchmarks. Guided answer generation is available, but without a strict inline citation format such as SRC_001. Conversation memory is also present, but it is only stored in memory and not in a persistent backend such as Redis.

Remaining Work

The main technical priorities are to add a persistent vector store for chunk embeddings, such as pgvector or Pinecone, add a proper BM25 retrieval layer, add an LLM safety classifier for ambiguous crisis-related cases, and integrate a real live hybrid retrieval flow combining graph retrieval and vector retrieval in parallel, followed by merging, deduplication, reranking, and context-budget trimming. Another important goal is to improve ranking quality for symptom-to-diagnosis-style answers.

Some work has been explicitly deferred for later stages, including strict inline citation formatting, advanced answer post-processing, post-generation safety scanning, persistent session storage, and more advanced conversation management.

Current Prototype Results

The project is already advanced enough to support thesis demonstrations, query a merged clinical graph, return answers grounded in explicit sources, and compare GraphRAG with vector baselines.

According to the current audit, PTSD, bipolar disorder, depression, and generalized anxiety disorder coverage are generally usable for demo purposes. However, some treatment relations remain too broad or too generic, and graph coverage is still selective rather than exhaustive.

The available proxy benchmark snapshot shows the following results. For GraphRAG, answer relevancy is 0.1651, context precision is 0.1261, context recall is 0.2993, and faithfulness is 0.8042. For the vector baseline, answer relevancy is 0.0906, context precision is 0.0423, context recall is 0.4608, and faithfulness is 0.9780.

These results suggest that GraphRAG currently performs better than the vector baseline in answer relevancy and context precision, while context recall remains lower because the graph is intentionally selective and still not exhaustive.

Current Backend State

The main live backend is Neo4j. The main API entry point is fastapi_app.py, and the main query pipeline is graphrag_pipeline.py. When available, the merged clinical and DSM checkpoint is used as the default graph source.

Scientific and Academic Value

This project demonstrates a complete applied research workflow, including corpus construction, document processing, knowledge extraction, graph representation, retrieval-augmented answering, application safety, and experimental evaluation.

It is a strong basis for a final-year thesis, an oral defense, a technical demonstration, and future research continuation.

Proposed Continuation Plan

In the short term, the next steps are to finalize the hybrid graph and vector pipeline, add BM25, add the LLM safety classifier, and strengthen the regression suite.

In the medium term, the project should improve relation quality and graph coverage, make provenance cleaner from end to end, and improve citation structuring in final generated answers.

In the long term, the approach can be extended to other medical subdomains, conversation memory can be made more robust, and the prototype can evolve into a stronger system for broader evaluation.

Short Presentation Version

This project evaluates a GraphRAG system for medical question answering. The prototype builds a knowledge graph from official mental-health sources and uses it to provide answers that are more grounded, more traceable, and safer than answers produced by a purely vector-based retrieval system.

A more formal presentation version would be as follows. We developed a GraphRAG prototype for mental-health question answering using official sources such as WHO, NIMH, and DSM-derived material. The system transforms documents into entities, relations, and cited supporting evidence, then queries this graph through an API. Current results show improved answer relevancy and context precision compared with a vector baseline, while also highlighting future work on graph coverage and hybrid retrieval.

Repository Sources Used for This Sheet

This technical sheet was built from the following project files: README.md, CURRENT_SYSTEM_OVERVIEW.md, WORK_PROGRESS.md, CLINICAL_GRAPH_AUDIT.md, poster_presentation_content.md, memoire/cover.tex, and paper/main.tex.
