"""Package entry point.

Set environment guards as early as possible so that downstream imports of
``transformers`` (via ``peft``, ``fastembed``, or any other path) see the
correct values regardless of which sub-module is imported first.
"""

import os

# Suppress the HuggingFace tokenizers fork-warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Skip transformers' optional TensorFlow integration. SmolLM2 + LoRA only
# need the PyTorch backend; some Conda-bundled tensorflow builds segfault on
# import under numpy 2.x. ``transformers.utils.import_utils`` reads ``USE_TF``
# when it first loads and caches the result, so this must be set before any
# transformers import — including transitive ones via peft or fastembed.
os.environ.setdefault("USE_TF", "0")
