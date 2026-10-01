import pickle
import numpy as np

embeddings = np.load("data/usda_embeddings.npy")

with open("data/usda_metadata.pkl","rb") as f:
    metadata= pickle.load(f)

def search_food(image_embeddings,top_k=5):
    """
    image embedding: numpy array of shape(1,512)
    """
    if image_embeddings.ndim==1:
        image_embeddings=image_embeddings.reshape(1,-1)
    # These vectors are normalized CLIP embeddings. The original FAISS index
    # used IndexFlatIP, so a NumPy dot product gives the same ranking while
    # avoiding an OpenMP runtime conflict between FAISS and PyTorch on macOS.
    similarities = embeddings @ image_embeddings[0]
    indices = np.argpartition(similarities, -top_k)[-top_k:]
    indices = indices[np.argsort(similarities[indices])[::-1]]
    results=[]
    for idx in indices:
        results.append({
            "Food code":metadata[idx]['Food code'],
            "Main food description":metadata[idx]['Main food description'],
            "score":float(similarities[idx])
        })

    return results
