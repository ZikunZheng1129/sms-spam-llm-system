# Qualitative Discussion of Output Quality

## Overview
This project compares four SMS-classification settings:

1. **Base LLM (No RAG)**
2. **Basic RAG**
3. **Advanced Agentic RAG with Base Model**
4. **Advanced Agentic RAG with Fine-Tuned LoRA Model**

Across the five test queries, the overall pattern is clear: retrieval generally improves evidence grounding, and the full agentic workflow with the base model produces the most stable and well-supported outputs in this setup. By contrast, the fine-tuned LoRA model runs successfully as an agent within the workflow, but its predictions are less reliable on several obvious spam examples.

---

## 1. Base LLM (No RAG)
The Base LLM produces generally reasonable classifications for the selected test queries. Its responses are concise and often correct, especially for clearly spam-like messages and straightforward personal messages. However, its outputs are based only on the user query itself, without any retrieved supporting examples. As a result, although the answers are often plausible, they are not explicitly grounded in external evidence from the dataset.

In this sense, the Base LLM serves as a useful baseline: it shows what the system can do without retrieval or agentic reasoning, but it does not provide the same level of interpretability as the RAG-based workflows.

---

## 2. Basic RAG
Basic RAG improves on the Base LLM by incorporating semantically similar retrieved SMS examples from the vector database. In the comparison outputs, this usually leads to explanations that are more evidence-aware. For example, spam predictions are justified not only by general wording patterns such as “prize” or “urgent,” but also by similarity to retrieved spam messages in the dataset.

Compared with the Base LLM, Basic RAG is more grounded and often more persuasive. However, it is still a relatively simple pipeline: retrieval happens once, and the model directly produces a final answer without any explicit evidence filtering or answer verification.

---

## 3. Advanced Agentic RAG with Base Model
The Advanced Agentic RAG with the base model is the strongest and most stable configuration in this project. It adds two important components beyond Basic RAG:

- an **evidence filter agent**, which removes less useful retrieved examples
- a **verification agent**, which checks whether the final prediction is supported by the selected evidence

This extra structure makes the system more interpretable and more robust. In the comparison results, this mode correctly classifies both spam and ham examples, and its outputs are consistently marked as **supported** by the verifier. The final responses are not only correct, but also explicitly grounded in the retrieved evidence and validated by a second reasoning step.

Among the four settings, this mode provides the best balance of correctness, interpretability, and workflow transparency.

---

## 4. Advanced Agentic RAG with Fine-Tuned LoRA Model
The Advanced Agentic RAG with the fine-tuned LoRA model is the most interesting setting because it demonstrates that simply introducing a fine-tuned model into an agentic workflow does not automatically improve final quality.

Technically, this mode works correctly:
- the LoRA adapter loads successfully
- it is integrated as one of the graph’s classifier agents
- it produces a usable label output
- it participates in the same retrieval, filtering, and verification workflow

However, the qualitative results show that the fine-tuned model is less reliable on several obvious spam examples. In multiple cases, it predicts **HAM** for messages that the other three settings classify as **SPAM**. Importantly, the verification agent frequently marks these LoRA-based outputs as **unsupported**, which shows that the surrounding agentic workflow is doing useful corrective work even when the classifier itself is weak.

This is an important project finding: the fine-tuned model is operationally integrated, but in this particular setup it does not outperform the stronger base-model agentic workflow. Instead, it highlights the value of the verification stage as a safeguard within an advanced RAG system.

---

## Overall Trend
The overall quality trend across the four settings can be summarized as follows:

- **Base LLM**: reasonable but ungrounded
- **Basic RAG**: more grounded and usually stronger than Base LLM
- **Advanced Agentic RAG with Base Model**: strongest overall, most stable, and well supported
- **Advanced Agentic RAG with Fine-Tuned LoRA Model**: fully integrated, but weaker in practice on several spam cases

This suggests that the largest improvement in this project comes not just from fine-tuning, but from the combination of:
- retrieval
- evidence selection
- structured multi-agent reasoning
- verification

---

## Conclusion
The comparison shows that retrieval and agentic workflow design significantly improve the interpretability and reliability of SMS classification. In this project, the **Advanced Agentic RAG with Base Model** performs best overall because it combines grounding, evidence selection, and answer verification in a stable way. The **LoRA-based advanced workflow** is still valuable as part of the stitching project because it demonstrates successful integration of a fine-tuned model into the agent graph, but its weaker performance also shows that fine-tuning alone does not guarantee better end-task behavior.