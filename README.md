# Food Nutrition Assistant

This project is now structured as an SDE-style REST application:

- `backend/` contains the Spring Boot REST API.
- `front.py` remains the Streamlit frontend.
- `backend/src/main/resources/nutrition-foods.csv` contains the USDA nutrition data used by the API.

## Run the Backend

From the project root:

```bash
cd backend
mvn spring-boot:run
```

The API runs on:

```text
http://127.0.0.1:8091
```

## Run the Frontend

In another terminal:

```bash
streamlit run front.py
```

## REST Endpoints

### Analyze Food

```http
POST /analyze-image
```

Form fields:

- `text`: optional food name.
- `file`: optional food image. The Spring Boot version uses the uploaded file name as a fallback search term when text is not provided.

### Confirm Food

```http
POST /confirm-food
```

JSON body:

```json
{
  "state_id": "confirmation-id",
  "food": "selected food name"
}
```

## Notes

The earlier FastAPI and AI pipeline files are still present for reference, but the active backend for this SDE-role version is the Spring Boot service inside `backend/`.
