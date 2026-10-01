import json
from embeddings.image_embeddings import embed_image
from retrieval.faiss_search import search_food
from usda.fetch_nutrition_excel import fetch_usda_nutrition_from_excel
from llm.mistral import llm, explain_nutrition_with_recipes


def embed_image_node(state):
    embedding = embed_image(state["image"])
    return {**state, "image_embedding": embedding}


def retrieve_foods(state):
    results = search_food(
        state["image_embedding"].astype("float32"),
        top_k=5
    )
    return {**state, "candidates": results}


def disambiguation_agent(state):
    candidates = state["candidates"]

    text = "\n".join(
        f"{c['Main food description']} (score={c['score']:.3f})"
        for c in candidates
    )

    prompt = f"""
You are a food identification agent.

Candidates:
{text}

If one food is clearly correct → return AUTO
Else → return ASK

AUTO:
{{"action":"AUTO","food":"<food name>"}}

ASK:
{{"action":"ASK","options":["food1","food2","food3"]}}
"""

    decision = json.loads(llm.invoke(prompt))

    if decision["action"] == "AUTO":
        selected = next(
            c for c in candidates
            if c["Main food description"] == decision["food"]
        )
        return {**state, "selected_food": selected}

    return {
        **state,
        "options": decision["options"]
    }


def nutrition_lookup(state):
    food_id = state["selected_food"]["Food code"]
    nutrition = fetch_usda_nutrition_from_excel(food_id)
    return {**state, "nutrition": nutrition}


def explain_nutrition(state):
    explanation = explain_nutrition_with_recipes(
        food=state["selected_food"],
        nutrition=state["nutrition"]
    )
    return {**state, "explanation": explanation}
