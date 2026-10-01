# pipeline.py

from embeddings.image_embeddings import embed_image
from embeddings.text_embeddings import embed_text
from retrieval.faiss_search import search_food
from usda.fetch_nutrition_excel import fetch_usda_nutrition_from_excel
from llm.mistral import explain_nutrition_with_recipes, llm
import json


def disambiguate_food(candidates):
    candidate_text = "\n".join(
        f"{c['Main food description']} (score={c['score']:.3f})"
        for c in candidates
    )

    prompt = f"""
You are a food identification decision agent.

Candidates:
{candidate_text}

Rules:
- If one food is clearly dominant, return AUTO
- If ambiguous, return ASK_USER
- Return ONLY valid JSON, nothing else.

AUTO format:
{{"action":"AUTO","food":"<food name>"}}

ASK_USER format:
{{"action":"ASK_USER","options":["food1","food2","food3"]}}
"""

    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        print(f"Mistral decision unavailable ({type(exc).__name__}); asking the user to choose.")
        return {
            "action": "ASK_USER",
            "options": [c["Main food description"] for c in candidates[:3]]
        }

    raw = response.content.strip()

    # 🛑 SAFETY CHECK
    if not raw.startswith("{"):
        print("⚠️ LLM returned invalid JSON:")
        print(raw)
        return {
            "action": "ASK_USER",
            "options": [c["Main food description"] for c in candidates[:3]]
        }

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️ JSON parse failed:")
        print(raw)
        return {
            "action": "ASK_USER",
            "options": [c["Main food description"] for c in candidates[:3]]
        }



def analyze_image_pipeline(image=None,text=None):
    """
    Main pipeline (NO LangGraph)
    """

    if image and text:
        img_emb=embed_image(image)
        txt_emb=embed_text(text)
        embedding=(img_emb+txt_emb)/2

    elif image:
        embedding=embed_image(image)

    elif text:
        embedding=embed_text(text)
    

    if embedding is None:
        return {"status": "error", "message": "Could not embed image"}

    candidates = search_food(embedding.astype("float32"), top_k=5)

    if not candidates:
        return {"status": "error", "message": "No food candidates found"}

    decision = disambiguate_food(candidates)

    if decision["action"] == "ASK_USER":
        return {
            "status": "need_confirmation",
            "options": decision["options"],
            "candidates": candidates,
        }

    # AUTO path
    selected = next(
        c for c in candidates
        if c["Main food description"] == decision["food"]
    )

    nutrition = fetch_usda_nutrition_from_excel(selected["Food code"])
    explanation = explain_nutrition_with_recipes(
        food=selected,
        nutrition=nutrition
    )

    return {
        "status": "done",
        "food": selected["Main food description"],
        "nutrition": nutrition,
        "explanation": explanation
    }
