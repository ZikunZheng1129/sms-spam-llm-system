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
| advanced_lora | 75 | 0.9600 | 0.9293 | 0.8923 | 0.9096 | 0.8000 | 0.9846 | 8 | 1 | 2 | 64 | 13.3135 | 0 |
| guarded_fallback | 75 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10 | 0 | 0 | 65 | 13.3192 | 0 |
