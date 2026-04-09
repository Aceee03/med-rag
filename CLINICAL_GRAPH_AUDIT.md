# Clinical Graph Audit

## Snapshot

- Source corpus: 10 official NIMH/WHO documents
- Parsed structural nodes: 137
- Clean graph checkpoint: `checkpoints/clinical_graph_clean`
- Neo4j sync verified: 207 nodes / 234 relationships

## Graph Quality Summary

- Strengths:
  - PTSD symptoms, treatment, prevalence, and comorbidity are useful and source-grounded.
  - Bipolar treatment, prevalence, and comorbidity coverage are strong enough for demo use.
  - Depression treatment coverage is good.
  - Autism prevalence and comorbidity coverage are good.
- Weaknesses:
  - Depression prevalence is missing.
  - Generalized anxiety prevalence is missing.
  - PTSD differential diagnosis is missing.
  - Generalized anxiety differential diagnosis is missing.
  - Autism treatment extraction is weak.
  - Several noisy entities still appear in answers, especially for bipolar and depression comparison queries.

## Manual Audit Matrix

| Condition | Symptoms | Treatment | Risk Factors | Prevalence | Differential | Comorbidity |
| --- | --- | --- | --- | --- | --- | --- |
| Depression | Good | Good | Weak | Missing | Mixed / noisy | Mixed |
| Generalized Anxiety Disorder | Good | Good | Mixed | Missing | Missing | Good |
| Autism Spectrum Disorder | Mixed / noisy | Weak | Mixed | Good | Mixed / noisy | Good |
| Bipolar Disorder | Mixed / noisy | Good | Mixed / noisy | Good | Thin | Good |
| PTSD | Good | Good | Mixed | Good | Missing | Good |

## Benchmark Snapshot

Proxy benchmark from `checkpoints/benchmark_clinical_clean_proxy/ragas_summary.md`:

| System | Status | Answer Relevancy | Context Precision | Context Recall | Faithfulness |
| --- | --- | --- | --- | --- | --- |
| vector_baseline | proxy | 0.0828 | 0.0392 | 0.4562 | 0.9833 |
| vector_reranked | proxy | 0.0855 | 0.0410 | 0.4728 | 0.9833 |
| graph_rag | proxy | 0.1629 | 0.1408 | 0.2924 | 0.7719 |

Interpretation:

- GraphRAG is materially better than the vector baselines on answer relevancy and context precision.
- Context recall is lower because the graph is sparser and some relation families are still missing.
- This supports a thesis claim that graph retrieval improves precision and groundedness, but not yet that it is fully comprehensive.

## Recommendation

The cleaned clinical graph is strong enough to use as the main thesis prototype and demo graph.

It is not yet polished enough to present as a final clinical-grade graph without qualification. The next quality pass should focus on:

1. Removing low-signal symptom / risk-factor nodes such as `DISABILITY`, `STIGMA`, `ABILITY TO FUNCTION`, and `VIRUSES`.
2. Improving prevalence extraction for depression and generalized anxiety disorder.
3. Improving differential diagnosis extraction for PTSD and generalized anxiety disorder.
4. Tightening autism treatment extraction so diagnostic evaluation is not mistaken for treatment.
5. Normalizing citation strings for PDF-derived sections.
