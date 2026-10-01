from typing import List,Dict, TypedDict, Optional


class FoodState(TypedDict):
    image: bytes
    image_embeddings:Optional[object]
    candidates: List[Dict]
    selected_food: Optional[Dict]
    options: Optional[List[str]]
    nutrition: Optional[Dict]
    explanation: Optional[str]
    route: Optional[str]


    