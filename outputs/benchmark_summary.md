# Benchmark Summary

_This benchmark uses a stratified held-out test sample. LLM/RAG modes may be run on smaller samples to control API cost._

## Run config

- canonical split: `test_size=0.2`, `seed=42`
- sample-size argument: 30 (effective: 30)
- random seed: 42
- spam ratio in sample: 0.133

## Results

| mode | n | acc | P_macro | R_macro | F1_macro | spam_recall | ham_recall | TP | FP | FN | TN | avg_latency_s | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf_lr | 30 | 0.9667 | 0.9815 | 0.8750 | 0.9191 | 0.7500 | 1.0000 | 3 | 0 | 1 | 26 | 0.0001 | 0 |
| base_llm | 30 | 0.8667 | 0.7292 | 0.8173 | 0.7600 | 0.7500 | 0.8846 | 3 | 3 | 1 | 23 | 0.8539 | 0 |
| basic_rag | 30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4 | 0 | 0 | 26 | 1.4625 | 0 |
| advanced_base | 30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4 | 0 | 0 | 26 | 5.6724 | 0 |
| advanced_lora | 30 | 0.8667 | 0.4333 | 0.5000 | 0.4643 | 0.0000 | 1.0000 | 0 | 0 | 4 | 26 | 11.0517 | 0 |
| guarded_fallback | 30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4 | 0 | 0 | 26 | 11.3166 | 0 |
