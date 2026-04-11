# Clinical Graph Audit

## Snapshot

- Source corpus: 10 official NIMH/WHO documents
- Parsed structural nodes: 137
- Clean graph checkpoint: `checkpoints/clinical_graph_clean`
- Neo4j sync / checkpoint snapshot: 200 nodes / 228 relationships

## Graph Quality Summary

- Strengths:
  - PTSD symptoms, treatment, prevalence, and comorbidity are useful and source-grounded.
  - Bipolar treatment, prevalence, and comorbidity coverage are strong enough for demo use.
  - Depression now has explicit prevalence coverage.
  - Generalized anxiety disorder now has explicit prevalence and differential-diagnosis coverage.
  - PTSD now has explicit differential-diagnosis coverage.
  - Autism now has useful treatment coverage in addition to prevalence and comorbidity.
- Weaknesses:
  - Some treatment nodes are still broad or programmatic rather than intervention-specific.
  - Bipolar prevalence / risk-factor coverage still contains a few generic graph nodes.
  - Comparison answers are much cleaner now, but they still mix high-value facts with some broad treatment labels.

## Manual Audit Matrix

| Condition | Symptoms | Treatment | Risk Factors | Prevalence | Differential | Comorbidity |
| --- | --- | --- | --- | --- | --- | --- |
| Depression | Good | Good | Weak | Good | Mixed | Mixed |
| Generalized Anxiety Disorder | Good | Good | Mixed | Good | Good | Good |
| Autism Spectrum Disorder | Mixed / noisy | Good | Mixed | Good | Mixed / noisy | Good |
| Bipolar Disorder | Mixed / noisy | Good | Mixed / noisy | Good | Thin | Good |
| PTSD | Good | Good | Mixed | Good | Good | Good |

## Benchmark Snapshot

Proxy benchmark from `checkpoints/benchmark_final_verify/ragas_summary.md`:

| System | Status | Answer Relevancy | Context Precision | Context Recall | Faithfulness |
| --- | --- | --- | --- | --- | --- |
| vector_baseline | proxy | 0.0906 | 0.0423 | 0.4608 | 0.9780 |
| vector_reranked | proxy | 0.0944 | 0.0437 | 0.4673 | 0.9789 |
| graph_rag | proxy | 0.1651 | 0.1261 | 0.2993 | 0.8042 |

Interpretation:

- GraphRAG remains materially better than the vector baselines on answer relevancy and context precision.
- Context recall is still lower because the graph is intentionally selective and relation coverage is not exhaustive.
- The current checkpoint is strong enough for thesis demos and targeted evaluation, with clearer citations and better repaired coverage than the previous snapshot.

## Recommendation

The cleaned clinical graph is now in a much better state for demo and thesis evaluation use.

It still needs qualification as a prototype rather than a clinical-grade system. The next quality pass should focus on:

1. Tightening broad treatment / support nodes in bipolar and depression answers.
2. Improving a few remaining broad risk-factor and prevalence nodes.
3. Expanding the regression suite as new conditions or source families are added.
