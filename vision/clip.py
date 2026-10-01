import torch
import io
from PIL import Image
from transformers import CLIPProcessor,CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

clip_model=CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)

clip_processor=CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

FOOD_NAMES=[
    "medu vada",
    "dosa",
    "idli",
    "chicken curry",
    "paneer curry"
]

FOOD_PROMPTS=[
    f"a photo of {food}" for food in FOOD_NAMES
]

with torch.no_grad():
    text_inputs=clip_processor(
        text=FOOD_PROMPTS,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    TEXT_EMBEDDINGS= clip_model.get_text_features(**text_inputs)
    TEXT_EMBEDDINGS= TEXT_EMBEDDINGS/TEXT_EMBEDDINGS.norm(
        dim=1, keepdim=True
    )


def classify_food(image_bytes:bytes,top_k:int=3,confidence_threshold:float=0.4):
    """
    Classify a cropped food image using CLIP.

    Returns:{
     food: str,
     confidence: float,
     alternatives: list[str],
     confidence_ok: bool
    }

    """
    image= Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with torch.no_grad():
        image_inputs= clip_processor(
            images=image,
            return_tensors="pt"
        ).to(DEVICE)
        image_embedding=clip_model.get_image_features(**image_inputs)
        image_embedding=image_embedding/image_embedding.norm(
            dim=1,keepdim=True
        )

    similarities= cosine_similarity(
        image_embedding.cpu().numpy(),
        TEXT_EMBEDDINGS.cpu().numpy()
    )[0]

    top_indices=similarities.argsort()[-top_k:][::-1]
    best_food=FOOD_NAMES[top_indices[0]]
    confidence= float(similarities[top_indices[0]])

    alternatives=[FOOD_NAMES[i] for i in top_indices[1:]]

    return{
        "food":best_food,
        "confidence":round(confidence,3),
        "alternatives":alternatives,
        "confidence_ok":confidence>=confidence_threshold
    }