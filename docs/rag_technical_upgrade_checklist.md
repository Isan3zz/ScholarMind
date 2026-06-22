# IRIS Paper Assistant RAG Technical Upgrade Checklist

This checklist focuses on turning IRIS into an engineering-grade paper reading and literature review assistant for large-company internship interviews, not a toy demo.

## Goal

Upgrade IRIS from a basic Agentic RAG demo into a measurable, explainable, and reproducible paper assistant.

Target positioning:

> A LangGraph-based Agentic RAG research system with structured document ingestion, hybrid retrieval, reranking, evidence-grounded generation, and RAG evaluation.

Recommended product positioning:

> IRIS: a LangGraph-based Agentic RAG paper reading and literature review assistant that supports paper PDF parsing, section-aware chunking, multi-paper retrieval, source-page citations, review generation, and iterative refinement.

## P0: High-Impact Core Upgrades

### 1. Paper PDF Parsing And Cleaning

- [ ] Normalize PDF text extracted by `PyPDFLoader`.
- [ ] Remove repeated headers, footers, page numbers, conference names, arXiv footers, copyright notices, and obvious boilerplate.
- [ ] Merge broken lines caused by PDF layout extraction.
- [ ] Fix broken words caused by line wrapping, such as `trans-\nformer`.
- [ ] Handle two-column extraction noise as much as possible with layout-aware post-processing.
- [ ] Preserve metadata for each text unit:
  - [ ] `source`
  - [ ] `page`
  - [ ] `paper_title`
  - [ ] `authors`
  - [ ] `section`
  - [ ] `subsection`
  - [ ] `chunk_type`
  - [ ] `paragraph_index`
  - [ ] `chunk_id`
- [ ] Detect and remove or down-weight low-value sections such as table of contents and references.
- [ ] Keep `Abstract`, `Introduction`, `Related Work`, `Method`, `Experiments`, `Results`, `Discussion`, `Conclusion`, `Limitations`, and `Appendix` as first-class section metadata.
- [ ] Preserve captions for figures and tables when available.
- [ ] Convert tables into Markdown-style text if table extraction is added later.

Resume angle:

> Designed a paper-specific PDF cleaning pipeline that removes layout noise while preserving title, author, page, section, subsection, and chunk-type metadata for explainable paper retrieval.

### 2. Paper Section Detection

- [ ] Implement a section detector for common paper sections:
  - [ ] `Abstract`
  - [ ] `1 Introduction`
  - [ ] `2 Related Work`
  - [ ] `3 Method`
  - [ ] `3.1 Model Architecture`
  - [ ] `4 Experiments`
  - [ ] `5 Results`
  - [ ] `6 Discussion`
  - [ ] `7 Conclusion`
  - [ ] `Limitations`
  - [ ] `Ethics Statement`
  - [ ] `Acknowledgements`
  - [ ] `References`
  - [ ] `Appendix`
- [ ] Normalize section aliases:
  - [ ] `Method`, `Methods`, `Approach`, `Framework`, `Model`
  - [ ] `Experiments`, `Evaluation`, `Results`
  - [ ] `Background`, `Preliminaries`
- [ ] Assign every paragraph to the nearest detected section.
- [ ] Mark unknown text with `section = "Unknown"` instead of dropping it.
- [ ] Use section metadata later for retrieval routing and filtering.

Resume angle:

> Built a paper section detector that identifies Abstract, Method, Experiments, Results, References, and Appendix sections, enabling section-aware retrieval and literature-review generation.

### 3. Paper-Aware Chunking

- [ ] Replace fixed large chunking with structure-aware chunking.
- [ ] Split documents first by section headings.
- [ ] Split sections by paragraphs.
- [ ] Split oversized paragraphs by token length.
- [ ] Use section-specific chunk sizes:
  - [ ] `Abstract`: keep as one complete chunk.
  - [ ] `Introduction` and `Related Work`: 500-800 tokens.
  - [ ] `Method`, `Experiments`, and `Results`: 300-600 tokens.
  - [ ] `Conclusion`: 300-600 tokens.
  - [ ] `References`: do not index by default, or index separately with low priority.
- [ ] Use semantic overlap based on sentence or paragraph boundaries instead of blind character overlap.
- [ ] Store neighbor links:
  - [ ] `prev_chunk_id`
  - [ ] `next_chunk_id`
- [ ] Add chunk statistics logging:
  - [ ] number of chunks per file
  - [ ] average chunk length
  - [ ] max chunk length
  - [ ] section distribution
- [ ] Preserve paper-specific chunk types:
  - [ ] `abstract`
  - [ ] `body`
  - [ ] `caption`
  - [ ] `table`
  - [ ] `formula`
  - [ ] `reference`

Resume angle:

> Implemented paper-aware chunking based on sections, paragraphs, and chunk types to reduce semantic fragmentation in long academic PDFs.

### 4. Figure, Table, And Formula Handling

- [ ] Detect figure captions such as `Figure 1:` or `Fig. 2:`.
- [ ] Detect table captions such as `Table 1:`.
- [ ] Keep captions as separate chunks with `chunk_type = "caption"`.
- [ ] Attach captions to nearby parent context.
- [ ] Preserve formulas as text when extraction quality is acceptable.
- [ ] Mark formula-heavy chunks with `chunk_type = "formula"`.
- [ ] If table extraction is added, convert tables into Markdown or structured text.
- [ ] Attach table chunks to their surrounding explanation paragraphs.
- [ ] Prioritize tables and result sections for experiment-related questions.

Resume angle:

> Preserved paper figures, tables, formulas, and captions as typed evidence chunks, improving retrieval for experimental results and method details.

### 5. Parent-Child Chunking

- [ ] Build small child chunks for retrieval.
- [ ] Build larger parent chunks for generation context.
- [ ] Store mapping from `child_chunk_id` to `parent_chunk_id`.
- [ ] Retrieve child chunks first.
- [ ] Expand selected child chunks into their parent contexts before sending to the writer.
- [ ] Deduplicate parent contexts before generation.
- [ ] Compare retrieval quality against the current flat chunking baseline.
- [ ] For papers, define parent chunks as:
  - [ ] one full subsection when short enough
  - [ ] a paragraph group within the same section
  - [ ] a table or figure caption plus surrounding explanation

Resume angle:

> Implemented Parent-Child chunking: small chunks improve recall precision, while parent contexts preserve enough evidence for faithful report generation.

### 6. Hybrid Retrieval

- [ ] Keep dense vector retrieval for semantic matching.
- [ ] Keep BM25 retrieval for keyword, formula, metric, and proper-noun matching.
- [ ] Use Elasticsearch as the unified retrieval backend.
- [ ] Add metadata filtering by:
  - [ ] session or thread
  - [ ] file
  - [ ] section
  - [ ] page range
  - [ ] chunk type
- [ ] Fuse BM25 and dense results with RRF.
- [ ] Log retrieval details for debugging:
  - [ ] query
  - [ ] retrieval mode
  - [ ] source file
  - [ ] page
  - [ ] raw rank
  - [ ] fused rank

Resume angle:

> Built a hybrid retrieval pipeline combining BM25 and dense embeddings with rank fusion to improve both terminology matching and semantic recall.

### 7. Section-Aware Retrieval Routing

- [ ] Classify user questions into paper-reading intents:
  - [ ] paper summary
  - [ ] method explanation
  - [ ] experiment result analysis
  - [ ] related work comparison
  - [ ] limitation analysis
  - [ ] citation/evidence lookup
  - [ ] multi-paper synthesis
- [ ] Route method questions to `Method`, `Approach`, `Model`, and `Framework`.
- [ ] Route experiment questions to `Experiments`, `Evaluation`, `Results`, tables, and captions.
- [ ] Route background questions to `Introduction`, `Related Work`, and `Background`.
- [ ] Route limitation questions to `Limitations`, `Discussion`, and `Conclusion`.
- [ ] Use routing as a soft metadata boost, not a hard filter, unless confidence is high.

Resume angle:

> Added section-aware retrieval routing for paper QA, boosting Method, Experiment, Related Work, or Limitation sections according to user intent.

### 8. Two-Stage Reranking

- [ ] Retrieve a wider candidate set, such as top 30.
- [ ] Deduplicate near-identical chunks before reranking.
- [ ] Use reranker to select top 5-8 evidence chunks.
- [ ] Preserve rerank scores in metadata.
- [ ] Add a relevance threshold.
- [ ] Trigger refusal or Web Search fallback when local relevance is below threshold.
- [ ] Return scores and sources to the frontend for explainability.

Resume angle:

> Designed a two-stage retrieval architecture with hybrid recall and cross-encoder reranking, plus confidence gating to reduce hallucinations from irrelevant documents.

## P1: Agentic Retrieval Improvements

### 9. Query Expansion

- [ ] Use the Planner node to generate subqueries for complex research questions.
- [ ] Retrieve using the original query and all subqueries.
- [ ] Add multi-query rewriting for different phrasings of the same intent.
- [ ] Add optional HyDE:
  - [ ] generate a hypothetical answer
  - [ ] retrieve using the hypothetical answer
  - [ ] merge with normal retrieval results
- [ ] Track which query retrieved each chunk.
- [ ] Add paper-specific query templates:
  - [ ] `What is the main contribution of this paper?`
  - [ ] `What method does the paper propose?`
  - [ ] `What datasets and metrics are used?`
  - [ ] `What are the main experimental results?`
  - [ ] `What are the limitations?`
  - [ ] `How does this paper differ from prior work?`

Resume angle:

> Used an agentic planner to decompose broad research questions into multiple retrieval intents, improving recall coverage for open-ended queries.

### 10. Evidence-Grounded Paper Generation

- [ ] Require the Writer node to cite sources for key claims.
- [ ] Format citations as `[source: file.pdf, p. 3]`.
- [ ] Pass source metadata together with chunk content to the LLM.
- [ ] Add a post-generation citation checker.
- [ ] If citations are missing, send the report back to the Reviewer or Refiner.
- [ ] Show evidence snippets in the frontend.
- [ ] Support paper-assistant output modes:
  - [ ] single-paper summary
  - [ ] method explanation
  - [ ] experiment table summary
  - [ ] literature review
  - [ ] multi-paper comparison
  - [ ] limitations and future work summary

Resume angle:

> Added evidence-grounded paper generation with source-page citations and post-generation citation checks for better answer traceability.

### 11. Local Knowledge Sufficiency Detection

- [ ] Replace simple YES/NO relevance grading with score-based sufficiency detection.
- [ ] Use signals from:
  - [ ] top rerank score
  - [ ] average top-k rerank score
  - [ ] number of distinct relevant sources
  - [ ] grader LLM judgment
- [ ] Define outcomes:
  - [ ] enough local evidence
  - [ ] weak local evidence, use hybrid mode
  - [ ] no local evidence, refuse or search web
- [ ] Log which rule triggered the decision.

Resume angle:

> Designed a local-knowledge sufficiency gate that decides whether to answer from documents, fallback to Web Search, or refuse unsupported questions.

## P2: Evaluation And Engineering Depth

### 9. RAG Evaluation Dataset

- [ ] Create `eval/dataset.jsonl`.
- [ ] Include 10-20 representative questions.
- [ ] Each sample should include:
  - [ ] question
  - [ ] expected source document
  - [ ] expected page or section
  - [ ] answer keywords
  - [ ] whether the system should refuse
- [ ] Include negative samples where uploaded documents are irrelevant.
- [ ] Include multi-document questions requiring synthesis.
- [ ] Include paper-specific tasks:
  - [ ] main contribution extraction
  - [ ] method retrieval
  - [ ] experiment result retrieval
  - [ ] limitation retrieval
  - [ ] cross-paper comparison
  - [ ] citation page accuracy

Resume angle:

> Built a RAG regression dataset covering normal retrieval, multi-document synthesis, and irrelevant-document refusal scenarios.

### 10. Retrieval Metrics

- [ ] Implement Recall@k.
- [ ] Implement MRR.
- [ ] Implement source hit rate.
- [ ] Implement citation accuracy.
- [ ] Compare multiple chunking strategies:
  - [ ] fixed-size chunking
  - [ ] paragraph chunking
  - [ ] structure-aware chunking
  - [ ] parent-child chunking
- [ ] Save results as a Markdown or CSV report.

Resume angle:

> Evaluated retrieval strategies with Recall@k, MRR, and citation accuracy, using metrics to guide chunking and reranking improvements.

### 11. Observability

- [ ] Add structured logs for ingestion, retrieval, reranking, and generation.
- [ ] Record timings for:
  - [ ] PDF parsing
  - [ ] embedding
  - [ ] ES indexing
  - [ ] hybrid retrieval
  - [ ] reranking
  - [ ] writing
  - [ ] reviewing
- [ ] Add request-level IDs.
- [ ] Add session-level IDs.
- [ ] Add failure reason logs.

Resume angle:

> Added structured observability for the RAG pipeline, tracking latency and decision paths across ingestion, retrieval, reranking, and generation.

### 12. Tests

- [ ] Test upload resets the knowledge base by default.
- [ ] Test file validation and unsafe filename handling.
- [ ] Test chunk metadata creation.
- [ ] Test retrieval result deduplication.
- [ ] Test refusal behavior on irrelevant documents.
- [ ] Test citation checker.
- [ ] Add CI for backend tests and frontend build.

Resume angle:

> Added regression tests for ingestion, retrieval, refusal logic, and citation validation to improve maintainability.

## Suggested Implementation Order

1. [ ] Paper PDF cleaning and metadata preservation.
2. [ ] Paper section detection.
3. [ ] Paper-aware chunking.
4. [ ] Figure, table, formula, and caption handling.
5. [ ] Parent-Child chunking.
6. [ ] Hybrid retrieval with RRF.
7. [ ] Section-aware retrieval routing.
8. [ ] Reranking with threshold-based sufficiency detection.
9. [ ] Evidence citation in generated paper summaries and literature reviews.
10. [ ] Paper-specific RAG evaluation dataset and metrics.
11. [ ] Observability and CI.

## Interview Storyline

Use this storyline when explaining the technical depth:

> I did not treat RAG as just embedding plus vector search. I optimized the pipeline specifically for academic papers. First, I cleaned PDF layout noise and preserved title, author, section, page, and chunk-type metadata. Then I added paper section detection and paper-aware parent-child chunking. For retrieval, I combined BM25 and dense embeddings with rank fusion, added section-aware routing for Method, Experiments, Related Work, and Limitations, then used reranking and a relevance threshold to decide whether local evidence was sufficient. Finally, I built a paper-specific evaluation set to compare chunking and retrieval strategies using Recall@k, MRR, citation accuracy, and refusal accuracy.
