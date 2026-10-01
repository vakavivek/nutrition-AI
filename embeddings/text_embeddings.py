import torch
import numpy as np
from transformers import CLIPModel,CLIPProcessor

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

model= CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor= CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_text(text:str)->np.ndarray:
    inputs=processor(
        text=[text],
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        text_features= model.get_text_features(**inputs)

    # Transformers 5 returns a model output object; older versions returned
    # the projected feature tensor directly.
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output

    text_features=text_features/text_features.norm(dim=-1,keepdim=True)
    return text_features.cpu().numpy()[0]
