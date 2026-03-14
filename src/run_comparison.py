from pathlib import Path
from src.evaluation import run_all_modes

QUERIES = [
    "Congratulations! You have won a free prize. Call now to claim.",
    "Hey, are we still meeting for lunch at 12 today?",
    "URGENT! Your mobile number has won £5000. Reply now to claim.",
    "Can you send me the notes from class when you get a chance?",
    "You have been specially selected for a cash reward. Call this number immediately.",
]

OUTPUT_PATH = Path("outputs/comparison_results.md")


def format_block(title: str, content: str) -> str:
    return f"### {title}\n\n{content.strip()}\n"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Comparison Results\n"]

    for i, query in enumerate(QUERIES, start=1):
        results = run_all_modes(query)

        lines.append(f"## Query {i}\n")
        lines.append(f"**Input:**  \n{query}\n")

        base_output = results["base_llm"]["raw_output"]
        basic_output = results["basic_rag"]["raw_output"]
        advanced_base_output = results["advanced_rag_base"]["final_answer"]
        advanced_lora_output = results["advanced_rag_lora"]["final_answer"]

        lines.append(format_block("Base LLM (No RAG)", base_output))
        lines.append(format_block("Basic RAG", basic_output))
        lines.append(format_block("Advanced Agentic RAG with Base Model", advanced_base_output))
        lines.append(format_block("Advanced Agentic RAG with Fine-Tuned LoRA Model", advanced_lora_output))
        lines.append("---\n")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparison results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
