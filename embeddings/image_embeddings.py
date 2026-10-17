import io

import numpy as np
import torch
from PIL import Image

from embeddings.clip_model import get_clip


def embed_image(image_input):
    if image_input is None:
        return None

    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif isinstance(image_input, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        raise ValueError(f"unsupported image type: {type(image_input)}")

    model, processor = get_clip()
    inputs = processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        features = model.get_image_features(**inputs)

    if hasattr(features, "pooler_output"):
        features = features.pooler_output

    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.cpu().numpy().astype(np.float32)
