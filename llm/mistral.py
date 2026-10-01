from langchain_mistralai.chat_models import ChatMistralAI
import os
from dotenv import load_dotenv

load_dotenv()

llm= ChatMistralAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.3,
    max_retries=1,
    timeout=30,
)

def explain_nutrition_with_recipes(food:str,nutrition:dict)->str:
    """
    Uses Mistral to explain nutrition and suggest recipe ideas
    """
    prompt=f"""
       You are a nutrition assistant.

       Food:{food}
       Nutrition per serving:
        calories: {nutrition.get('calories')}
        Protein:{nutrition.get('protein')}
        carbs:{nutrition.get('carbs')}
        Fat:{nutrition.get('fat')}

       Tasks:
       1. Explain this nutrition in simple, non-medical language
       2. Suggest 2 healthy recipe or preparation ideas
       3. Do not calculate or modify nutrition numbers
       4. keep it consise (4-6 lines).
"""
    try:
        return llm.invoke(prompt).content
    except Exception as exc:
        print(f"Mistral explanation unavailable ({type(exc).__name__}); using a local summary.")
        food_name = (
            food.get("Main food description", "This food")
            if isinstance(food, dict)
            else str(food)
        )
        return (
            f"Per 100 g, {food_name} provides {nutrition.get('calories', 'N/A')} kcal, "
            f"{nutrition.get('Protein', 'N/A')} g protein, "
            f"{nutrition.get('Carbohydrates', 'N/A')} g carbohydrates, and "
            f"{nutrition.get('Fats', 'N/A')} g fat. "
            "For a meal idea, serve it with vegetables and whole grains, or add it to a balanced bowl or wrap."
        )
