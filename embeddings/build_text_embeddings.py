import pandas as pd
import numpy as np
import torch
import pickle
from transformers import CLIPProcessor, CLIPModel

DEVICE= "cuda" if torch.cuda.is_available() else "cpu"

model_name="openai/clip-vit-base-patch32"

model= CLIPModel.from_pretrained(model_name).to(DEVICE)
processor= CLIPProcessor.from_pretrained(model_name)

df=pd.read_excel('../data/usda_foods.xlsx',header=1)

texts= df['Main food description'].astype(str).tolist()

embeddings=[]
BATCH_SIZE=32

for i in range(0,len(texts),BATCH_SIZE):
    batch= texts[i:i+BATCH_SIZE]

    inputs= processor(
        text=batch,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        text_features= model.get_text_features(**inputs)

    text_features= text_features/text_features.norm(dim=-1,keepdim=True)

    embeddings.append(text_features.cpu().numpy())

embeddings=np.vstack(embeddings).astype("float32")

np.save("../data/usda_embeddings.npy",embeddings)
metadata=df[['Food code','Main food description']].to_dict(orient="records")
with open("../data/usda_metadata.pkl","wb") as f:
    pickle.dump(metadata,f)


print("Text Embeddings created",embeddings.shape)



