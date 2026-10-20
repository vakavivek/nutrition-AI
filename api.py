from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE_STORE = {}


@app.get("/health")
def health():
    return {"status": "ok", "service": "nutrition-ai-api"}


@app.post("/analyze-image")
async def analyze_image(file: UploadFile | None = File(None), text: str | None = Form(None)):
    image_bytes = await file.read() if file else None

    from pipeline import analyze_image_pipeline

    result = analyze_image_pipeline(image=image_bytes, text=text)

    if result["status"] == "need_confirmation":
        state_id = str(uuid.uuid4())
        STATE_STORE[state_id] = result

        # Include candidate options with their similarity scores for transparency
        formatted_options = [
            {"food": c["Main food description"], "score": float(c.get("score", 0.0))}
            for c in result.get("candidates", [])[:3]
        ]

        return {
            "status": "need_confirmation",
            "state_id": state_id,
            "options": formatted_options
        }

    return result


@app.post("/confirm-food")
def confirm_food(payload: dict):
    state_id = payload.get("state_id")
    chosen_food = payload.get("food")

    if not state_id or state_id not in STATE_STORE:
        return {"status": "error", "message": "Session expired or invalid state. Please analyze the image again."}

    state = STATE_STORE.pop(state_id)

    candidates = state.get("candidates", [])
    selected = next(
        (c for c in candidates if c.get("Main food description") == chosen_food),
        None
    )

    if not selected and candidates:
        selected = candidates[0]

    if not selected:
        return {"status": "error", "message": "Selected food candidate could not be found."}

    try:
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
            "score": float(selected.get("score", 0.0)),
            "nutrition": nutrition,
            "explanation": explanation
        }
    except Exception as exc:
        return {"status": "error", "message": f"Backend processing error: {str(exc)}"}