from embeddings.image_embeddings import embed_image
from retrieval.faiss_search import search_food

image_embedding= embed_image("apple1.jpeg")
results= search_food(image_embedding,top_k=5)

print("\nTop Matches:\n")
for r in results:
    print(f"{r['Main food description']} | score={r['score']:.3f}")