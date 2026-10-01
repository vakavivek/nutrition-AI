import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io 


DEVICE="cuda" if torch.cuda.is_available() else "cpu"

model_name="openai/clip-vit-base-patch32"

model= CLIPModel.from_pretrained(model_name).to(DEVICE)
processor= CLIPProcessor.from_pretrained(model_name)

def embed_image(image_input):
    if image_input is None:
        return None
    if isinstance(image_input,Image.Image):
        image= image_input.convert("RGB")
    elif isinstance(image_input,(bytes,bytearray)):
        image= Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input,str):
        image = Image.open(image_input).convert("RGB")
    else:
        raise ValueError(f"unsupported image type:{type(image_input)}")
    inputs= processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        image_features= model.get_image_features(**inputs)

    # Transformers 5 returns a model output object; older versions returned
    # the projected feature tensor directly.
    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output

    image_features= image_features/image_features.norm(dim=-1,keepdim=True)
    return image_features.cpu().numpy()
