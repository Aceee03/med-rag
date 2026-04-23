# Short Presentation for Poster Session

Suggested length: 4 to 5 minutes total

Main framing: a general presentation of the GraphRAG system, with the mental-health application used as a prototype case study

## Slide 1 - Title and Main Idea

### Text on the slide

**GraphRAG for Trustworthy Question Answering**

- Goal: improve grounding and reduce hallucinations
- Core idea: retrieve from a knowledge graph instead of relying only on free generation
- Prototype application: mental-health question answering

### Illustration

- Clean title slide with:
- a small graph visual
- document icons feeding into a network
- a short subtitle such as: *Structured retrieval, source grounding, and safer answers*

### Gemini illustration prompt

Create a clean light 2D flat title-slide illustration on a pure white background. Use only blue and black text, thin blue lines, and simple flat vector shapes. Show several document icons on the left feeding into a central blue node-link graph, with a clean answer card or chatbot bubble on the right. The style should feel academic, modern, minimal, and polished. Add visual space for a presentation title at the top or center. No gradients, no shadows, no 3D, no clutter, 16:9 composition.

### What to say

This project presents a GraphRAG system designed to make question answering more trustworthy. The main idea is to combine large language models with structured graph-based retrieval, so the answer is grounded in explicit entities, relations, and source evidence. We then validated this system through a prototype application in the mental-health domain.

## Slide 2 - What Is GraphRAG?

### Text on the slide

**GraphRAG in one sentence**

- GraphRAG = Retrieval-Augmented Generation built on a knowledge graph
- Documents are transformed into entities, relations, and connected evidence
- The system retrieves structured graph context for the question
- The LLM generates an answer grounded in that retrieved structure

### Illustration

- A simple concept diagram:
- Documents -> Knowledge graph -> Graph retrieval -> Grounded answer
- Show a few connected nodes and labeled edges in the middle

### What to say

GraphRAG stands for Retrieval-Augmented Generation using a knowledge graph. Instead of retrieving only text chunks, the system organizes information as entities and relations, then retrieves connected evidence from that structure. This means the model answers using an explicit representation of how concepts are linked, which makes the result easier to ground and explain.

### Gemini illustration prompt

Create a clean light 2D flat infographic on a pure white background. Use only blue and black text, thin blue connector lines, and simple flat vector icons. Show a left-to-right flow: document icons on the left, a central knowledge graph with blue nodes and labeled black edges, and a chatbot answer card on the right. Add short labels: "Documents", "Knowledge Graph", "Graph Retrieval", and "Grounded Answer". Modern academic presentation style, minimal, airy layout, no gradients, no 3D, no shadows, no clutter, 16:9 composition.

## Slide 3 - Traditional RAG vs GraphRAG

### Text on the slide

**Two retrieval paradigms**

- Traditional RAG:
- retrieves similar text chunks
- strong for local semantic matching
- weaker for explicit relationships and multi-hop reasoning

- GraphRAG:
- retrieves entities, relations, and graph neighborhoods
- stronger for connected reasoning and comparisons
- more traceable through graph structure and citations

### Illustration

- A side-by-side comparison:
- Left column: Traditional RAG with query -> retrieved chunks -> answer
- Right column: GraphRAG with query -> graph nodes and edges -> answer with connected evidence

### What to say

Traditional RAG and GraphRAG both try to ground the model with retrieved information, but they do it differently. Traditional RAG mainly retrieves text chunks that look semantically similar to the query. GraphRAG retrieves structured evidence such as entities, relations, and local graph neighborhoods. This is especially useful when the question depends on relationships, comparisons, or multi-step connections between concepts.

### Gemini illustration prompt

Design a clean 2D flat comparison infographic on a white background with blue and black text only. Split the image into two balanced vertical panels. Left panel title: "Traditional RAG" with a query box leading to stacked text chunks and then an answer box. Right panel title: "GraphRAG" with a query box leading to a small blue node-link graph and then an answer box with citation markers. Use flat vector shapes, thin blue lines, black labels, minimal academic style, lots of white space, no gradients, no shadows, no 3D, 16:9 layout.

## Slide 4 - Graph Construction Pipeline

### Text on the slide

**Offline pipeline**

1. Ingest source documents
2. Convert documents to markdown
3. Split them into structured chunks
4. Enrich each chunk with context
5. Extract entities and relations with an LLM
6. Clean, repair, and merge the graph
7. Save graph checkpoints for runtime use

### Illustration

- A pipeline diagram with arrows:
- Sources -> Markdown -> Chunks -> Context enrichment -> Entity/relation extraction -> Graph cleanup -> Checkpoint

### Gemini illustration prompt

Create a clean 2D flat process infographic on a white background with blue and black text only. Show a left-to-right pipeline with seven simple labeled steps: "Sources", "Markdown", "Chunks", "Context Enrichment", "Entity & Relation Extraction", "Graph Cleanup", and "Checkpoint". Use thin blue arrows, simple document and graph icons, flat vector shapes, and a minimal academic style. Keep the layout airy and balanced. No gradients, no shadows, no 3D, 16:9 format.

### What to say

The offline pipeline transforms raw documents into a queryable graph. Documents are first converted into markdown, then split by structure such as headings and paragraphs. Each chunk is enriched with contextual information, and an LLM extracts entities and relations from it. After that, the graph is cleaned, repaired, and saved as a checkpoint, so the runtime system can load it efficiently.

## Slide 5 - Runtime Architecture

### Text on the slide

**Online query flow**

1. User sends a question
2. Safety layer checks the input
3. Query is classified
4. Graph retrieval finds relevant entities and subgraphs
5. System returns an answer, citations, and exact source chunks

- Query types:
- entity lookup
- symptom or feature lookup
- comparison
- treatment lookup

### Illustration

- Horizontal system diagram:
- User -> Safety shield -> Query classifier -> Graph retrieval -> Answer + citations

### Gemini illustration prompt

Design a clean light 2D flat architecture graphic on a pure white background using only blue and black text. Show a horizontal flow with five blocks: "User Question", "Safety Layer", "Query Classifier", "Graph Retrieval", and "Answer + Citations". Connect them with thin blue arrows. Use simple icons for user, shield, routing, graph, and answer card. Minimal academic presentation style, lots of white space, flat vector design, no gradients, no shadows, no 3D, 16:9 layout.

### What to say

At runtime, the system starts with a safety check, then classifies the question and chooses the right retrieval path. It can support different query types such as direct entity lookup, reverse lookup from symptoms or features, comparisons, and treatment-related questions. The output includes not only an answer, but also citations and the exact source chunks that support it.

## Slide 6 - Why This Design Is Useful

### Text on the slide

**Expected advantages of GraphRAG**

- Better reasoning over relationships between concepts
- More targeted retrieval than chunk similarity alone
- Easier traceability through graph paths and citations
- Better control when the system should abstain or qualify an answer

### Illustration

- A benefit map with four labeled blocks:
- Structure
- Precision
- Traceability
- Safety

### Gemini illustration prompt

Create a clean flat 2D benefits infographic on a white background with blue and black text only. Place four balanced blocks or cards labeled "Structure", "Precision", "Traceability", and "Safety" around a small central graph icon. Use thin blue connecting lines and simple academic vector styling. Make it minimal, elegant, and presentation-ready with lots of white space. No gradients, no shadows, no 3D, 16:9 composition.

### What to say

This design is useful because it gives structure to the retrieval process. Instead of treating every passage as isolated text, the system can reason through relationships such as symptoms, treatments, or comparisons between concepts. That usually leads to more targeted retrieval, clearer provenance, and better control over when the system has enough evidence to answer confidently.

## Slide 7 - Prototype Case Study: Mental Health

### Text on the slide

**Prototype application**

- Domain used for validation: mental-health question answering
- Sources: official WHO and NIMH documents plus DSM-derived material
- Clinical source corpus: 10 official NIMH/WHO documents
- Final merged prototype graph: **2,719 entities** and **3,918 relations**

### Illustration

- Small domain-specific graph snippet with nodes such as:
- Depression
- PTSD
- Anxiety
- Symptoms
- Treatments

### Gemini illustration prompt

Create a clean light 2D flat case-study illustration on a white background with blue and black text only. Show a small knowledge graph focused on a mental-health prototype, with blue nodes labeled "Depression", "PTSD", "Anxiety", "Symptoms", and "Treatments" connected by thin blue lines. Keep it simple, academic, and non-clinical in tone. No realistic people, no medical drama, no gradients, no shadows, no 3D, 16:9 composition.

### What to say

To validate the system, we applied it to a mental-health question-answering prototype. This domain is a good test case because it requires both factual grounding and careful handling. We built the graph from authoritative sources including WHO, NIMH, and DSM-derived material. The final merged prototype graph contains 2,719 entities and 3,918 relations.

## Slide 8 - Prototype Results and What We Learned

### Text on the slide

**Prototype findings**

- GraphRAG outperformed vector-only baselines on answer relevancy
- It also improved context precision
- Benchmark snapshot:
- Answer relevancy: **0.165** vs **0.091** for vector baseline
- Context precision: **0.126** vs **0.042**
- Context recall: **0.299** vs **0.461**

**Interpretation**

- GraphRAG was more selective and more precise
- Recall remained lower because graph coverage is not exhaustive

### Illustration

- Small bar chart comparing:
- answer relevancy
- context precision
- context recall

### Gemini illustration prompt

Design a clean 2D flat benchmark slide graphic on a pure white background using blue and black text only. Show a minimal grouped bar chart comparing "GraphRAG" and "Vector Baseline" across three metrics: "Answer Relevancy", "Context Precision", and "Context Recall". Use blue bars for GraphRAG and black or light gray bars for the baseline, with clean labels and lots of white space. Academic presentation style, flat vector design, no gradients, no shadows, no 3D, 16:9 layout.

### What to say

In the prototype evaluation, GraphRAG performed better than the vector baseline on answer relevancy and context precision. This suggests that graph retrieval was able to bring back more focused and useful evidence. At the same time, context recall was lower, which reflects a known tradeoff: the graph is selective, but its coverage is not yet complete. So the prototype shows that GraphRAG is promising, while also highlighting where improvement is still needed.

## Slide 9 - Conclusion and Future Work

### Text on the slide

**Conclusion**

- GraphRAG is a promising framework for grounded question answering
- The system combines graph structure, source evidence, and safer retrieval
- The mental-health prototype shows the approach is practical and testable

**Future work**

- expand graph coverage
- improve relation quality
- strengthen hybrid retrieval
- evaluate on broader domains

### Illustration

- Roadmap visual:
- Current system -> stronger graph -> hybrid retrieval -> broader evaluation

### Gemini illustration prompt

Create a clean light 2D flat roadmap infographic on a white background with blue and black text only. Show a horizontal progression with four stages labeled "Current System", "Stronger Graph", "Hybrid Retrieval", and "Broader Evaluation". Use thin blue arrows, simple flat icons, and a modern academic presentation style with generous white space. No gradients, no shadows, no 3D, 16:9 composition.

### What to say

To conclude, the main contribution is the GraphRAG system itself: a pipeline that builds a graph from trusted documents and uses it to support grounded question answering. The mental-health prototype serves as an example showing that the approach is feasible and useful. The next step is to improve coverage, refine relation quality, and evaluate the same architecture on larger and more general domains.

## Optional Slide 10 - Closing

### Text on the slide

**Thank you**

- Questions?

### Illustration

- Reuse the poster background or a clean graph image

### Gemini illustration prompt

Create a minimal clean 2D flat closing slide illustration on a white background using blue and black only. Show a subtle blue knowledge-graph motif or a few connected nodes in one corner, leaving large clean space for a "Thank you" and "Questions?" message. Modern academic presentation style, elegant, airy, no gradients, no shadows, no 3D, 16:9 layout.

### What to say

Thank you for your attention. I’d be happy to discuss the graph construction pipeline, the runtime architecture, or the prototype evaluation in more detail.

## Quick Delivery Tips

- Spend most of the time on slides 2 to 6, since those present the system itself
- Present slides 7 and 8 as validation, not as the full identity of the project
- On the results slide, say clearly that these are prototype benchmark values
- If you need a shorter version, merge slides 7 and 8 into one case-study slide
