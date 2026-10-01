import faiss 
import numpy as np

embeddings= np.load("../data/usda_embeddings.npy").astype("float32")

dim= embeddings.shape[1]

index= faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index,"../data/usda_faiss.index")

print("FAISS index built with", index.ntotal, "vectors")