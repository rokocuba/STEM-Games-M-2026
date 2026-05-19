from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = Path("tf-logreg/reddit_ai_detector.joblib")

app = FastAPI(title="Reddit AI Text Detector")


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    label: str
    score: float
    probabilities: dict[str, float]
    reason: str


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


model = load_model()


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    text = req.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    prediction = model.predict([text])[0]

    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=500,
            detail="Loaded model does not support predict_proba",
        )

    probabilities_array = model.predict_proba([text])[0]
    classes = model.named_steps["classifier"].classes_

    probabilities = {
        label: float(prob)
        for label, prob in zip(classes, probabilities_array)
    }

    score = probabilities.get(str(prediction), 0.0)

    return {
        "label": str(prediction),
        "score": score,
        "probabilities": probabilities,
        "reason": "TF-IDF + Logistic Regression model",
    }
