# SMS Spam LLM System: Benchmarking Classical NLP, RAG, Agentic Verification, and LoRA Fine-Tuning

This repository is a benchmarked SMS spam classification system built on the UCI SMS Spam Collection dataset. It compares classical machine learning, direct LLM classification, retrieval-augmented generation, agentic RAG, a local LoRA-adapted classifier, and a guarded fallback workflow.

The project is designed as a portfolio-friendly LLM systems study: it includes implementation code in `src/`, cost-controlled quantitative benchmark outputs in `outputs/`, and analysis notebooks in `notebooks/`.

## Key Features

- **TF-IDF + Logistic Regression baseline** for a strong, fast classical NLP reference point.
- **Base LLM classification** for direct spam/ham prediction without retrieval.
- **Basic RAG** using Pinecone retrieval to ground classification in similar labeled SMS examples.
- **Advanced Agentic RAG** with evidence retrieval, evidence filtering, classification, verification, retry/finalization logic, and structured orchestration.
- **Evidence-aware LoRA classifier** using the local `smollm2_spam_lora_adapter/` as an experimental small-model classifier.
- **Guarded fallback mode** that runs LoRA first, checks verifier support, and falls back to the advanced base RAG workflow when needed.
- **Risk-feature logging** for explainability and error analysis, including URL, phone, currency, urgency, prize, call/reply, uppercase, exclamation, and length features.
- **Quantitative benchmark pipeline** that writes reproducible CSV, Markdown, and confusion-matrix artifacts.
- **Analysis notebooks** for dataset EDA, benchmark analysis, and prediction-level system insights.

## Project Architecture

```text
data/SMSSpamCollection
        |
        v
src/classical_baselines.py  ---> TF-IDF + Logistic Regression
        |
        v
src/benchmark.py -----------> runs selected modes, logs predictions, metrics, risk features
        |
        +--> src/base_modes.py --------> Base LLM and Basic RAG
        +--> src/pinecone_utils.py ----> Pinecone retrieval utilities
        +--> src/agents.py / graph.py -> Agentic retrieval, filtering, classification, verification
        +--> src/lora_utils.py -------> Local LoRA classifier wrapper
        +--> src/evaluation.py -------> Advanced RAG and guarded fallback runners
        |
        v
outputs/
        |
        v
notebooks/
```

The implementation stays in `src/`. The notebooks are analysis/reporting layers that read from `data/` and `outputs/`; they do not call APIs or rerun model inference.

## Modes Compared

| Mode | Uses retrieval? | Uses LLM? | Uses LoRA? | Uses verifier/fallback? | Notes |
|---|---|---|---|---|---|
| `tfidf_lr` | No | No | No | No | Classical TF-IDF + Logistic Regression baseline. |
| `base_llm` | No | Yes | No | No | Direct API-based LLM classification. |
| `basic_rag` | Yes | Yes | No | No | Retrieves similar labeled examples and passes them to the LLM. |
| `advanced_base` | Yes | Yes | No | Verifier/retry | Agentic RAG workflow with base LLM classifier. |
| `advanced_lora` | Yes | Yes | Yes | Verifier/retry | Agentic RAG workflow with local LoRA classifier and LLM-based evidence filtering/verifier. |
| `guarded_fallback` | Yes | Yes | Yes | Yes | Runs LoRA first, then falls back to advanced base RAG if verification is unsupported. |

## Benchmark Results

The table below comes from a **75-sample stratified held-out benchmark**. The sample was intentionally limited to control API cost across LLM/RAG modes. These results should be read as a cost-controlled comparison slice, not as full-test-set production validation.

| mode | n | accuracy | macro F1 | spam recall | ham recall | TP | FP | FN | TN | avg latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tfidf_lr` | 75 | 0.9867 | 0.9699 | 0.9000 | 1.0000 | 9 | 0 | 1 | 65 | 0.0001 |
| `base_llm` | 75 | 0.8800 | 0.7967 | 0.9000 | 0.8769 | 9 | 8 | 1 | 57 | 0.9988 |
| `basic_rag` | 75 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10 | 0 | 0 | 65 | 1.4964 |
| `advanced_base` | 75 | 0.9867 | 0.9723 | 1.0000 | 0.9846 | 10 | 1 | 0 | 64 | 5.9119 |
| `advanced_lora` | 75 | 0.8800 | 0.5585 | 0.1000 | 1.0000 | 1 | 0 | 9 | 65 | 10.7867 |
| `guarded_fallback` | 75 | 0.9867 | 0.9723 | 1.0000 | 0.9846 | 10 | 1 | 0 | 64 | 11.7479 |

Generated benchmark artifacts:

- `outputs/benchmark_results.csv`
- `outputs/benchmark_predictions.csv`
- `outputs/benchmark_summary.md`
- `outputs/confusion_matrices.png`

## Key Findings

- Retrieval improved LLM reliability on the 75-sample benchmark slice.
- TF-IDF + Logistic Regression remained a strong and extremely fast classical baseline.
- The base LLM was weaker without retrieval grounding, mainly due to false positives.
- Basic RAG and Advanced Base performed strongly on this benchmark sample.
- The LoRA adapter is integrated and runnable, but it still showed weak spam recall (`0.1000`) in this run.
- Evidence-aware prompting alone did not fix the LoRA classifier's ham bias.
- Guarded fallback reached `1.0000` spam recall, rescuing all 9 spam examples missed by LoRA.
- Fallback was used on 11 of 75 rows: 9 spam rows and 2 ham rows.
- System design matters as much as model choice: a small fine-tuned model does not automatically outperform RAG or a classical baseline.

## How to Run

### Install

```bash
pip install -r requirements.txt
```

### Environment

Create a `.env` file with the required credentials for API-backed modes:

```env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=sms-spam-bge384
```

The offline classical baseline does not require OpenAI or Pinecone credentials.

### Run CLI

```bash
python -m src.app
```

### Run Classical Baseline

```bash
python -m src.classical_baselines
```

### Run Offline Benchmark

```bash
python -m src.benchmark --modes tfidf_lr --sample-size 200
```

### Run Small All-Mode Benchmark

This command calls API-backed modes and should be used intentionally:

```bash
python -m src.benchmark --modes tfidf_lr,base_llm,basic_rag,advanced_base,advanced_lora,guarded_fallback --sample-size 75
```

### Run Qualitative Demo

```bash
python -m src.run_comparison
```

## Notebook Structure

- `notebooks/archive/` contains older development notebooks from the original project stages.
- `notebooks/01_dataset_eda.ipynb` summarizes the dataset and explainability risk features.
- `notebooks/02_baseline_and_benchmark_analysis.ipynb` analyzes benchmark metrics and confusion matrices.
- `notebooks/03_error_analysis_and_system_insights.ipynb` analyzes prediction-level errors, LoRA failures, guarded fallback corrections, and risk-feature patterns.

The notebooks do not call OpenAI, Pinecone, or LoRA models. They read existing files from `data/` and `outputs/`. Source implementation remains in `src/`.

## Repository Structure

```text
sms-spam-llm-system/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── SMSSpamCollection
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_baseline_and_benchmark_analysis.ipynb
│   ├── 03_error_analysis_and_system_insights.ipynb
│   └── archive/
│       ├── 01_vector_db_and_embeddings.ipynb
│       ├── 02_rag_baselines.ipynb
│       ├── 03_prompt_engineering.ipynb
│       └── 04_lora_finetuning.ipynb
├── src/
│   ├── __init__.py
│   ├── agents.py
│   ├── app.py
│   ├── base_modes.py
│   ├── benchmark.py
│   ├── classical_baselines.py
│   ├── config.py
│   ├── evaluation.py
│   ├── features.py
│   ├── graph.py
│   ├── lora_utils.py
│   ├── pinecone_utils.py
│   ├── prompts.py
│   └── run_comparison.py
├── outputs/
│   ├── baseline_tfidf_lr.joblib
│   ├── benchmark_predictions.csv
│   ├── benchmark_results.csv
│   ├── benchmark_summary.md
│   ├── confusion_matrices.png
│   ├── comparison_results.md
│   ├── discussion.md
│   ├── graph.png
│   └── sample_runs.md
└── smollm2_spam_lora_adapter/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── chat_template.jinja
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── README.md
```

## Limitations

- The all-mode benchmark shown above uses 75 held-out samples because LLM/RAG calls cost money.
- Results should be validated on a larger held-out test set before any production use.
- The current LoRA adapter has poor spam recall on the benchmark slice.
- Evidence-aware prompting alone did not fix the LoRA classifier's tendency to predict `ham`.
- Full RAG and agentic modes require valid OpenAI and Pinecone configuration.
- The LoRA loader uses a tokenizer fallback because the saved adapter tokenizer config contains a stale tokenizer class. The adapter itself is left unchanged.
- The benchmark is useful for comparing system behavior, but it is not a substitute for broader evaluation, calibration, and robustness testing.

## Future Work

- Run a larger benchmark over more of the held-out test set.
- Retrain or improve the LoRA adapter with stronger spam recall objectives.
- Calibrate verifier and fallback thresholds.
- Add a local vector store option for fully offline RAG experiments.
- Add CI tests for data loading, feature extraction, baseline metrics, and benchmark output schema.
- Add an optional web UI for interactive comparison across modes.

## Security and Reproducibility

- `.env` is ignored by `.gitignore`; secrets should never be committed.
- Benchmark outputs are generated artifacts and should not contain API keys or other credentials.
- Use environment variables or a local `.env` file for credentials.
- Do not commit OpenAI keys, Pinecone keys, adapter-private data, or local logs.

## Skills Demonstrated

- Classical NLP baselines with scikit-learn
- LLM application design
- Retrieval-augmented generation
- Vector database integration
- Agentic verification and fallback design
- Parameter-efficient fine-tuning integration with LoRA
- Quantitative benchmarking and error analysis
- Notebook-based reporting for ML systems

## Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection dataset
- Open-source tooling used for embeddings, retrieval, fine-tuning, and orchestration
