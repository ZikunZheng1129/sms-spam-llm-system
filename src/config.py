from dotenv import load_dotenv
import os
from pathlib import Path

# Suppress Hugging Face tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env at project startup
load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sms-spam-bge384")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K_RETRIEVAL = 5
MAX_RETRIES = 1

DEFAULT_TEMPERATURE = 0

BASE_HF_MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# change to yours
LORA_ADAPTER_PATH = str(Path(__file__).resolve().parent.parent / "smollm2_spam_lora_adapter")
