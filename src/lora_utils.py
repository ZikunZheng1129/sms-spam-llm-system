import re
import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.config import BASE_HF_MODEL_NAME, LORA_ADAPTER_PATH


@lru_cache(maxsize=1)
def load_lora_model():
    """
    Load the base Hugging Face model and attach the LoRA adapter.
    Cached so the model is only loaded once per Python process.
    """
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_HF_MODEL_NAME,
        dtype=torch.float32,
    )

    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.eval()

    return tokenizer, model


def build_lora_question(user_query: str) -> str:
    """
    Build lora question.
    """
    return f"""Classify the following SMS as spam or ham.

SMS: {user_query}

Answer with exactly one word: spam or ham."""


def generate_lora_response(user_query: str, max_new_tokens: int = 20) -> str:
    """
    Run generation with the LoRA fine-tuned model.
    """
    tokenizer, model = load_lora_model()

    question = build_lora_question(user_query)
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


def extract_lora_label(output_text: str) -> str:
    """
    Extract lora label.
    """
    matches = re.findall(r"\b(spam|ham)\b", output_text.lower())
    return matches[-1] if matches else "unknown"


def parse_lora_output(output_text: str) -> dict:
    """
    Return a normalized structure for downstream use.
    The LoRA model is used as a classifier, so explanation is left empty here.
    """
    label = extract_lora_label(output_text)

    return {
        "label": label,
        "explanation": "",
        "raw_output": output_text,
    }