"""LoRA classifier wrapper for the agentic SMS spam pipeline.

The LoRA adapter at ``LORA_ADAPTER_PATH`` is attached to the base
``BASE_HF_MODEL_NAME`` model (SmolLM2-1.7B-Instruct) on first call and cached
for the lifetime of the process via ``functools.lru_cache``. To compare
different adapter paths, run each adapter in a separate Python process so the
cached model cannot be reused across configurations.

Evidence-aware prompting (Phase 1, v2):
    ``build_lora_question`` and ``generate_lora_response`` accept an optional
    pre-formatted ``evidence_text`` block. When provided, the prompt includes a
    short list of labeled retrieved examples before the user query so the LoRA
    classifier can use the same evidence the agentic graph already retrieved.
    Callers without evidence (``evidence_text=None``) keep the original
    behavior, so existing call-sites stay valid.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

# Skip transformers' optional TensorFlow integration BEFORE importing peft /
# transformers. Some Conda-bundled tensorflow builds segfault on import under
# numpy 2.x; we only need the PyTorch backend for SmolLM2 + LoRA.
os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.config import BASE_HF_MODEL_NAME, LORA_ADAPTER_PATH  # noqa: E402


@lru_cache(maxsize=1)
def load_lora_model():
    """Load the base HF model and attach the LoRA adapter (cached per process).

    ``LORA_ADAPTER_PATH`` is read from ``src.config`` at import time and the
    loaded model is cached. CLI benchmark runs are separate Python processes,
    so setting ``LORA_ADAPTER_PATH=...`` per command is the intended way to
    compare adapters.

    The local adapter folder may have been saved by a PEFT version that wrote
    a ``tokenizer_class`` value (e.g. ``"TokenizersBackend"``) that newer
    transformers does not recognize. When that happens we fall back to the
    base model's tokenizer, which uses the same vocabulary so the LoRA
    forward pass remains correct.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)
    except (ValueError, OSError, KeyError) as exc:
        print(
            f"  load_lora_model: tokenizer load from {LORA_ADAPTER_PATH} "
            f"failed ({type(exc).__name__}: {exc}); "
            f"falling back to base model tokenizer."
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_HF_MODEL_NAME)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_HF_MODEL_NAME,
        dtype=torch.float32,
    )

    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.eval()

    return tokenizer, model


def format_lora_evidence(
    docs: list[dict[str, Any]] | None,
    max_examples: int = 4,
    max_chars: int = 200,
) -> str | None:
    """Format retrieved docs into a compact evidence block for the LoRA prompt.

    Caps to ``max_examples`` documents and truncates each to ``max_chars`` to
    keep the prompt short — small LMs degrade fast on over-long context.
    Returns ``None`` when there is nothing usable so the caller falls back to
    the no-evidence prompt.
    """
    if not docs:
        return None
    lines: list[str] = []
    for doc in docs[:max_examples]:
        label = (doc.get("label") or "?").strip().lower()
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        lines.append(f"- [{label}] {text}")
    if not lines:
        return None
    return "\n".join(lines)


def build_lora_question(
    user_query: str,
    evidence_text: str | None = None,
) -> str:
    """Build the LoRA classification prompt, optionally with retrieved evidence."""
    if evidence_text:
        return (
            "Classify the following SMS as spam or ham.\n\n"
            "Reference examples:\n"
            f"{evidence_text}\n\n"
            f"SMS: {user_query}\n\n"
            "Answer with exactly one word: spam or ham."
        )
    return (
        "Classify the following SMS as spam or ham.\n\n"
        f"SMS: {user_query}\n\n"
        "Answer with exactly one word: spam or ham."
    )


def generate_lora_response(
    user_query: str,
    evidence_text: str | None = None,
    max_new_tokens: int = 8,
) -> str:
    """Run generation with the LoRA fine-tuned model.

    Returns the newly generated text only (not the prompt echo) when possible,
    falling back to the full decoded sequence otherwise. The downstream
    ``extract_lora_label`` already takes the last spam/ham token, which is safe
    in both cases.
    """
    tokenizer, model = load_lora_model()

    question = build_lora_question(user_query, evidence_text=evidence_text)
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(input_text, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][prompt_len:]
    decoded_new = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if decoded_new.strip():
        return decoded_new
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_lora_label(output_text: str) -> str:
    """Extract the spam/ham label from the model output, defaulting to 'unknown'."""
    matches = re.findall(r"\b(spam|ham)\b", output_text.lower())
    return matches[-1] if matches else "unknown"


def parse_lora_output(output_text: str) -> dict[str, Any]:
    """Return a normalized structure for downstream use."""
    label = extract_lora_label(output_text)
    return {
        "label": label,
        "explanation": "",
        "raw_output": output_text,
    }
