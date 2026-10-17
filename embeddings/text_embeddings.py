import numpy as np
import torch

from embeddings.clip_model import get_clip


def embed_text(text: str) -> np.ndarray:
    model, processor = get_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)

    with torch.inference_mode():
        features = model.get_text_features(**inputs)

    if hasattr(features, "pooler_output"):
        features = features.pooler_output

    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.cpu().numpy()[0].astype(np.float32)
