from dotenv import load_dotenv
import os
from pathlib import Path

# Suppress Hugging Face tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Skip transformers' optional TensorFlow integration. SmolLM2 + LoRA only need
# the PyTorch backend; some Conda-bundled tensorflow builds segfault on import
# under numpy 2.x, so disabling the integration here prevents an unrelated
# crash from killing the LoRA pipeline.
os.environ.setdefault("USE_TF", "0")

# Load environment variables from .env at project startup
load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sms-spam-bge384")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K_RETRIEVAL = 5
MAX_RETRIES = 1

DEFAULT_TEMPERATURE = 0

BASE_HF_MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LORA_ADAPTER_PATH = REPO_ROOT / "smollm2_spam_lora_adapter_v2"

_lora_adapter_path = Path(
    os.getenv("LORA_ADAPTER_PATH", str(DEFAULT_LORA_ADAPTER_PATH))
)
if not _lora_adapter_path.is_absolute():
    _lora_adapter_path = REPO_ROOT / _lora_adapter_path
LORA_ADAPTER_PATH = str(_lora_adapter_path)
