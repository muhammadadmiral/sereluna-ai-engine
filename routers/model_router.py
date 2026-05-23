from fastapi import APIRouter

from services.nlp_service import evaluate_supervised_emotion_model
from services.nlp.ml_service import reload_supervised_emotion_model
from services.nlp.intent_service import evaluate_intent_model, reload_intent_model

router = APIRouter(prefix="/api/v1/model", tags=["model"])


@router.get("/emotion/evaluation/")
async def read_emotion_model_evaluation():
    return {
        "model": "TF-IDF + Logistic Regression Emotion Classifier",
        "evaluation": evaluate_supervised_emotion_model(),
    }


@router.post("/emotion/reload/")
async def reload_emotion_model():
    return {
        "model": "TF-IDF + Logistic Regression Emotion Classifier",
        "evaluation": reload_supervised_emotion_model(),
        "message": "Emotion model cache cleared and retrained from CSV dataset.",
    }


@router.get("/intent/evaluation/")
async def read_intent_model_evaluation():
    return {
        "model": "TF-IDF + ComplementNB Intent Classifier",
        "evaluation": evaluate_intent_model(),
    }


@router.post("/intent/reload/")
async def reload_intent_classifier():
    return {
        "model": "TF-IDF + ComplementNB Intent Classifier",
        "evaluation": reload_intent_model(),
        "message": "Intent model cache cleared and retrained from CSV dataset.",
    }
