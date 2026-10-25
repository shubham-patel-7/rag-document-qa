## Retrieval evaluation (25 questions, 6 documents, 30 chunks)

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR | Context recall@5 | Hit@3 (factoid) | Hit@3 (paraphrase) |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.92 | 0.96 | 0.96 | 0.94 | 0.96 | 1.00 | 0.91 |
| TF-IDF dense (FAISS) | 0.92 | 0.92 | 0.96 | 0.93 | 0.96 | 1.00 | 0.82 |
| Hybrid: TF-IDF + BM25 (RRF) | 0.92 | 0.96 | 0.96 | 0.93 | 0.96 | 1.00 | 0.91 |
| MiniLM dense (FAISS) | 0.88 | 0.96 | 1.00 | 0.93 | 1.00 | 1.00 | 0.91 |
| Hybrid: MiniLM + BM25 (RRF) | 0.92 | 1.00 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 |

### Answer check (hybrid retrieval, top-5)

| Retriever | Answerer | Answer contains all gold keywords | Answer cites the gold page |
|---|---|---|---|
| Hybrid: TF-IDF + BM25 (RRF) | extractive | 0.60 | 0.88 |
| Hybrid: MiniLM + BM25 (RRF) | extractive | 0.80 | 1.00 |

### Chunk-size sweep (hybrid retrieval)

| Embedder | Chunk words | #chunks | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|---|
| TF-IDF | 60 | 64 | 0.88 | 0.92 | 0.90 |
| TF-IDF | 120 | 30 | 0.92 | 0.96 | 0.93 |
| TF-IDF | 200 | 19 | 0.92 | 0.92 | 0.94 |
| MiniLM | 60 | 64 | 0.84 | 1.00 | 0.92 |
| MiniLM | 120 | 30 | 0.92 | 1.00 | 0.95 |
| MiniLM | 200 | 19 | 0.96 | 0.96 | 0.97 |
