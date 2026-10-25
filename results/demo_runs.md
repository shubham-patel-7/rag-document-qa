| Demo run | Embedder | Gold page rank (BM25 / dense / hybrid) | Answer | Cites gold page | Latency |
|---|---|---|---|---|---|
| 1. Factoid | tfidf | rank 1 / rank 1 / rank 1 | The Controller Area Network (CAN) protocol was developed by Bosch and introduced in 1986. [1] | yes | 2 ms |
| 2. Paraphrase | tfidf | rank 6 / not in top 10 / rank 6 | When either counter exceeds 127 the node becomes error-passive, and when the transmit error counter exceeds 25 | no | 2 ms |
| 3. Not in the documents | tfidf | n/a (no gold page) | A diesel engine is a compression-ignition engine. [1] Diesel Engine Diagnostics Basics: How a diesel engine wo | n/a | 2 ms |
| 1. Factoid | minilm | rank 1 / rank 1 / rank 1 | The Controller Area Network (CAN) protocol was developed by Bosch and introduced in 1986. [1] | yes | 195 ms |
| 2. Paraphrase | minilm | rank 6 / rank 1 / rank 3 | When either counter exceeds 127 the node becomes error-passive, and when the transmit error counter exceeds 25 | yes | 132 ms |
| 3. Not in the documents | minilm | n/a (no gold page) | The engine fuel rate is SPN 183 in bytes 1 and 2 with a resolution of 0.05 litres per hour per bit. [4] A dies | n/a | 172 ms |
