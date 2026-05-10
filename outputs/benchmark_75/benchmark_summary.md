# Benchmark Summary

_This benchmark uses a stratified held-out test sample. LLM/RAG modes may be run on smaller samples to control API cost._

## Run config

- canonical split: `test_size=0.2`, `seed=42`
- sample-size argument: 75 (effective: 75)
- random seed: 42
- spam ratio in sample: 0.133

## Results

| mode | n | acc | P_macro | R_macro | F1_macro | spam_recall | ham_recall | TP | FP | FN | TN | avg_latency_s | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf_lr | 75 | 0.9867 | 0.9924 | 0.9500 | 0.9699 | 0.9000 | 1.0000 | 9 | 0 | 1 | 65 | 0.0001 | 0 |
| base_llm | 75 | 0.8800 | 0.7561 | 0.8885 | 0.7967 | 0.9000 | 0.8769 | 9 | 8 | 1 | 57 | 0.9988 | 0 |
| basic_rag | 75 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10 | 0 | 0 | 65 | 1.4964 | 0 |
| advanced_base | 75 | 0.9867 | 0.9545 | 0.9923 | 0.9723 | 1.0000 | 0.9846 | 10 | 1 | 0 | 64 | 5.9119 | 0 |
| advanced_lora | 75 | 0.8800 | 0.9392 | 0.5500 | 0.5585 | 0.1000 | 1.0000 | 1 | 0 | 9 | 65 | 10.7867 | 0 |
| guarded_fallback | 75 | 0.9867 | 0.9545 | 0.9923 | 0.9723 | 1.0000 | 0.9846 | 10 | 1 | 0 | 64 | 11.7479 | 0 |
