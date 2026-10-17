"""Shared, lazily loaded CLIP model to avoid loading duplicate copies in RAM."""

import os
import threading

import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

# Keep CPU thread pools small on memory-constrained hosts.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(1)

_model = None
_processor = None
_load_lock = threading.Lock()


def get_clip():
    """Load one shared CLIP model only when the first embedding is requested."""
    global _model, _processor

    if _model is None or _processor is None:
        with _load_lock:
            if _model is None or _processor is None:
                # CPU is intentional: Render's small free instance has no CUDA GPU.
                model = CLIPModel.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
                model.to("cpu")
                model.eval()
                processor = CLIPProcessor.from_pretrained(MODEL_NAME)
                _model = model
                _processor = processor

    return _model, _processor
