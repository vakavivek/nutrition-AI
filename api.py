# api.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uuid

from pipeline import analyze_image_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE_STORE = {}


@app.post("/analyze-image")
async def analyze_image(file: UploadFile | None = File(None),text:str|None = Form(None)):
    image_bytes = await file.read() if file else None

    result = analyze_image_pipeline(image=image_bytes,text=text)

    if result["status"] == "need_confirmation":
        state_id = str(uuid.uuid4())
        STATE_STORE[state_id] = result

        return {
            "status": "need_confirmation",
            "state_id": state_id,
            "options": result["options"]
        }

    return result


@app.post("/confirm-food")
def confirm_food(payload: dict):
    state_id = payload["state_id"]
    chosen_food = payload["food"]

    state = STATE_STORE.pop(state_id)

    selected = next(
        c for c in state["candidates"]
        if c["Main food description"] == chosen_food
    )

    from usda.fetch_nutrition_excel import fetch_usda_nutrition_from_excel
    from llm.mistral import explain_nutrition_with_recipes

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
