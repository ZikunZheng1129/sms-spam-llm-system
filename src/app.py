from src.base_modes import run_base_llm, run_basic_rag
from src.evaluation import run_advanced_rag, run_all_modes


def main():
    print("=== SMS RAG Comparison System ===")
    print("Choose a mode:")
    print("1. base")
    print("2. basic_rag")
    print("3. advanced_base")
    print("4. advanced_lora")
    print("5. all")

    mode = input("Enter mode: ").strip().lower()
    user_query = input("Enter an SMS message: ").strip()

    if not user_query:
        print("No input provided. Exiting.")
        return

    if mode == "1" or mode == "base":
        result = run_base_llm(user_query)
        print("\n=== Base LLM ===")
        print(result["raw_output"])

    elif mode == "2" or mode == "basic_rag":
        result = run_basic_rag(user_query)
        print("\n=== Basic RAG ===")
        print(result["raw_output"])

    elif mode == "3" or mode == "advanced_base":
        result = run_advanced_rag(user_query, classifier_mode="base")
        print("\n=== Advanced Agentic RAG (Base Model) ===")
        print(result["final_answer"])

    elif mode == "4" or mode == "advanced_lora":
        result = run_advanced_rag(user_query, classifier_mode="lora")
        print("\n=== Advanced Agentic RAG (LoRA Model) ===")
        print(result["final_answer"])

    elif mode == "5" or mode == "all":
        results = run_all_modes(user_query)

        print("\n=== Base LLM ===")
        print(results["base_llm"]["raw_output"])

        print("\n=== Basic RAG ===")
        print(results["basic_rag"]["raw_output"])

        print("\n=== Advanced Agentic RAG (Base Model) ===")
        print(results["advanced_rag_base"]["final_answer"])

        print("\n=== Advanced Agentic RAG (LoRA Model) ===")
        print(results["advanced_rag_lora"]["final_answer"])

    else:
        print("Invalid mode. Please choose one of: base, basic_rag, advanced_base, advanced_lora, all.")


if __name__ == "__main__":
    main()
