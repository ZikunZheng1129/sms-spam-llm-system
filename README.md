# SMS Spam LLM System

A unified LAG-based and fine-tuning-based SMS spam classification project built on the **UCI SMS Spam Collection** dataset. This repository compares multiple increasingly advanced approaches, including **semantic retrieval**, **retrieval-augmented generation (RAG)**, **prompt engineering**, **LoRA fine-tuning**, and **multi-agent reasoning**.

## Project Overview

This project studies how different large language model enhancement strategies affect **SMS spam detection**. Rather than treating spam classification as a single-model problem, this repository builds a unified experimental framework that compares multiple approaches under the same dataset and task setting.

The project progresses through several stages:

- **Semantic retrieval and vector search** for nearest-neighbor example lookup
- **Basic RAG pipelines** for evidence-grounded classification
- **Prompt engineering strategies** for improving zero-shot and few-shot performance
- **LoRA fine-tuning** for parameter-efficient task adaptation
- **Multi-agent orchestration** for advanced evidence filtering, classification, and verification

The final system integrates these components into a modular framework for comparing direct LLM classification, RAG-based classification, and agentic reasoning workflows, while also including a fine-tuned LoRA branch as an experimental component.

## Research Question

**How do retrieval, prompt engineering, fine-tuning, and agent-based orchestration affect the quality and interpretability of LLM-based SMS spam classification?**

## Dataset

This project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository:

- **Dataset:** SMS Spam Collection
- **Task:** Binary text classification
- **Labels:** `spam` and `ham`

Source: UCI Machine Learning Repository  
Dataset link: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Methods

This repository compares multiple methods built on the same dataset and classification task.

### 1. Semantic Retrieval and Vector Database

The first stage builds a semantic retrieval layer by embedding SMS messages and storing them in a vector database. This enables nearest-neighbor search and forms the foundation for retrieval-based classification.

Main ideas:
- Convert SMS messages into dense embeddings
- Store and index embeddings in Pinecone
- Retrieve semantically similar messages for a query SMS
- Use retrieved examples as supporting evidence

### 2. Retrieval-Augmented Generation (RAG)

The second stage uses retrieval results to support LLM predictions. Instead of classifying a message in isolation, the model can reference similar labeled examples during inference.

Main ideas:
- Retrieve semantically similar messages
- Pass retrieved context into the prompt
- Generate a spam/ham prediction with explanation
- Compare retrieval-augmented predictions against baseline LLM outputs

### 3. Prompt Engineering

This stage evaluates how prompt design affects model behavior and classification performance.

Prompting strategies explored include:
- Baseline prompts
- Best-practice instruction prompts
- Few-shot prompts
- Prompt optimization workflows

This stage highlights how prompt structure alone can influence decision quality, explanation clarity, and robustness.

### 4. LoRA Fine-Tuning

This stage adapts a compact language model to the SMS spam classification task using **Low-Rank Adaptation (LoRA)**.

Main ideas:
- Start from a pretrained instruct model
- Fine-tune efficiently with LoRA
- Compare base and fine-tuned behavior
- Evaluate whether task-specific adaptation improves spam detection

This branch is included as an **experimental fine-tuned classifier**. In the current version of the project, the LoRA pipeline is fully integrated and runnable, but its spam recall is limited and it should not be interpreted as the strongest-performing method in this repository.

### 5. Multi-Agent Reasoning System

The final stage integrates retrieval, filtering, classification, and verification into an advanced multi-agent workflow.

The agentic system supports:
- Retrieving candidate evidence
- Filtering or selecting relevant supporting examples
- Performing classification
- Verifying or refining the final decision

This creates a more structured reasoning pipeline than direct prompting or basic RAG alone.

## Final System Modes

The final project compares several operating modes within one framework:

- **Base LLM:** direct classification without retrieval
- **Basic RAG:** retrieval-augmented classification using similar examples
- **Advanced Agentic RAG (Base Model):** multi-agent workflow with the base model
- **Advanced Agentic RAG (LoRA Model):** multi-agent workflow with the fine-tuned classifier

This makes the repository a unified comparative framework rather than a collection of isolated assignments.

## Current Status

The repository has been tested end to end in a clean Python 3.11 virtual environment.

Verified working components:
- Base LLM mode
- Basic RAG mode
- Pinecone retrieval and index connectivity
- Advanced agentic RAG with the base model
- Advanced agentic RAG with the LoRA model
- Full multi-mode comparison flow

Current behavior observed during testing:
- The **base**, **basic RAG**, and **advanced base** modes correctly classify obvious spam examples
- The **LoRA branch runs successfully**, but it shows weak spam recall and can misclassify obvious spam messages as `ham`

Because of this, the LoRA branch is best interpreted as an **experimental fine-tuning extension**, while the retrieval-based and agentic-base workflows are currently the most reliable components of the system.

## Repository Structure

```text
sms-spam-llm-system/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── SMSSpamCollection
├── notebooks/
│   ├── 01_vector_db_and_embeddings.ipynb
│   ├── 02_rag_baselines.ipynb
│   ├── 03_prompt_engineering.ipynb
│   └── 04_lora_finetuning.ipynb
├── src/
│   ├── agents.py
│   ├── app.py
│   ├── base_modes.py
│   ├── config.py
│   ├── evaluation.py
│   ├── graph.py
│   ├── lora_utils.py
│   ├── pinecone_utils.py
│   ├── prompts.py
│   └── run_comparison.py
├── outputs/
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

## Key Components

### `notebooks/`

Contains the method development stages that document the evolution of the project:

- vector search and embedding experiments
- retrieval-based spam classification
- prompt engineering experiments
- LoRA fine-tuning workflow

### `src/`

Contains the modular production-style system used for the final integrated pipeline:

- app entry point
- retrieval utilities
- prompts
- agent logic
- graph workflow
- evaluation and comparison scripts

### `outputs/`

Stores final diagrams, result summaries, and comparison artifacts.

### `smollm2_spam_lora_adapter/`

Contains the LoRA adapter produced during fine-tuning.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ZikunZheng1129/sms-spam-llm-system
cd sms-spam-llm-system
```

### 2. Create and activate a virtual environment

Using Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file and add the required credentials.

Example:

```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=sms-spam-bge384
OPENAI_API_KEY=your_key_here
```

### 5. Run the application

```bash
python -m src.app
```

### 6. Run comparisons or evaluation

```bash
python -m src.run_comparison
```

## Example Workflow

A typical end-to-end workflow in this project is:

1. Load and preprocess the SMS Spam Collection dataset  
2. Embed training messages and index them in a vector database  
3. Retrieve similar examples for a new SMS query  
4. Classify the message using:
   - direct prompting
   - RAG
   - agentic RAG
   - agentic RAG with a LoRA-adapted model  
5. Compare outputs across modes in terms of label quality and explanation quality

## Why This Project Matters

Spam detection is a well-defined classification problem, but this project goes beyond a standard classifier. It examines how modern LLM system design choices affect practical decision-making on a real text classification task.

This makes the repository useful as both:

- a **machine learning engineering project**
- a **comparative LLM systems study**

The broader goal is not only to classify SMS messages accurately, but also to understand how retrieval, prompting, fine-tuning, and multi-step reasoning interact in a unified pipeline.

## Results and Discussion

The project is designed to compare:

- direct LLM classification vs retrieval-supported classification
- prompt-only improvement vs task-specific fine-tuning
- simple single-step inference vs structured multi-agent reasoning

From the current tested system behavior:

- **Base LLM** correctly identifies obvious spam examples
- **Basic RAG** improves grounding by using retrieved similar messages
- **Advanced Agentic RAG (Base Model)** provides the strongest current combination of prediction quality and evidence-supported reasoning
- **Advanced Agentic RAG (LoRA Model)** is fully integrated and operational, but currently underperforms on spam recall

In the fine-tuning notebook, the saved LoRA adapter showed slightly higher overall accuracy than the base model on a very small test set, but both models failed on the available spam examples. In practice, this means the current fine-tuned model appears biased toward predicting `ham`, so its evaluation should be interpreted cautiously.

The repository therefore emphasizes both **system design** and **honest comparison across methods**, rather than claiming that every branch outperforms the others.

## Limitations

The current project has several important limitations:

- The LoRA branch is integrated successfully, but its spam recall is limited in its current form
- The fine-tuning evaluation in the notebook used a very small test split, so the reported metrics are not strong evidence of robust generalization
- The advanced LoRA workflow currently uses the fine-tuned classifier inside the agentic pipeline, but retrieved evidence is not directly incorporated into the LoRA generation prompt itself
- External services such as Pinecone and API-based LLM providers require environment configuration and valid credentials

These limitations make the current repository best viewed as a **working comparative system and engineering prototype**, with clear opportunities for more rigorous evaluation and model improvement.

## Future Work

There are several natural extensions to this project:

- Retrain the LoRA classifier with a stronger evaluation setup and a larger held-out test set
- Add quantitative benchmark tables across all modes
- Evaluate precision, recall, F1-score, and calibration
- Improve spam recall for the fine-tuned model
- Expand the agentic pipeline with more robust evidence ranking
- Pass retrieved evidence directly into the LoRA classification prompt
- Test additional embedding models and vector databases
- Explore larger or instruction-specialized fine-tuned models
- Add a user-facing web interface for interactive classification
- Extend the framework to other short-text safety or moderation tasks

## Skills Demonstrated

This project demonstrates experience with:

- Large language model application design
- Retrieval-augmented generation (RAG)
- Vector databases and embeddings
- Prompt engineering
- Parameter-efficient fine-tuning with LoRA
- Multi-agent workflow design
- Python-based ML system organization
- Experimental comparison across multiple model strategies
- Environment setup and end-to-end system validation

## Notes

This repository is the result of combining multiple assignment stages into one coherent project with a unified task, dataset, and system architecture. The final structure is intended to present the work as a single professional portfolio project rather than a collection of separate coursework submissions.

## Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection dataset
- Open-source tooling used for embeddings, retrieval, fine-tuning, and orchestration
