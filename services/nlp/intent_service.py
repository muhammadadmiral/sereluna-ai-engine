import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

from services.nlp.utils import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INTENT_DATASET = PROJECT_ROOT / "data" / "training" / "intent_dataset.csv"
INTENT_CONFIDENCE_THRESHOLD = 0.48
INTENT_CLASS_THRESHOLDS = {
    "emotional_support": 0.35,
    "casual_reference": 0.40,
    "casual_banter": 0.45,
}


def _read_intent_dataset() -> List[Dict[str, str]]:
    if not INTENT_DATASET.exists():
        return []
    with INTENT_DATASET.open("r", encoding="utf-8", newline="") as file:
        rows = [
            {"text": (row.get("text") or "").strip(), "intent": (row.get("intent") or "").strip()}
            for row in csv.DictReader(file)
        ]
    return [row for row in rows if row["text"] and row["intent"]]


def _intent_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 3),
                                lowercase=True,
                                sublinear_tf=True,
                                min_df=1,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(2, 5),
                                lowercase=True,
                                sublinear_tf=True,
                                min_df=1,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                ComplementNB(alpha=0.2),
            ),
        ]
    )


@lru_cache(maxsize=1)
def get_intent_model() -> Dict[str, Any]:
    rows = _read_intent_dataset()
    if not rows:
        return {"model": None, "dataset_size": 0, "classes": []}
    texts = [row["text"] for row in rows]
    intents = [row["intent"] for row in rows]
    model = _intent_pipeline()
    model.fit(texts, intents)
    return {
        "model": model,
        "dataset_size": len(rows),
        "classes": list(model.classes_),
    }


def classify_intent_supervised(text: str) -> Dict[str, Any]:
    normalized = normalize_text(text)
    bundle = get_intent_model()
    model = bundle.get("model")
    if not normalized or model is None:
        return {
            "intent": "reflective_companion",
            "confidence": 0.0,
            "accepted": False,
            "top_probabilities": [],
            "dataset_size": bundle.get("dataset_size", 0),
        }

    probabilities = model.predict_proba([normalized])[0]
    classes = list(model.classes_)
    ranked = sorted(
        (
            {"intent": str(intent), "probability": round(float(probability), 4)}
            for intent, probability in zip(classes, probabilities)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    top = ranked[0]
    confidence = float(top["probability"])
    threshold = INTENT_CLASS_THRESHOLDS.get(top["intent"], INTENT_CONFIDENCE_THRESHOLD)
    return {
        "intent": top["intent"],
        "confidence": round(confidence, 4),
        "accepted": confidence >= threshold,
        "top_probabilities": ranked[:3],
        "dataset_size": bundle["dataset_size"],
        "algorithm": {
            "name": "TF-IDF ComplementNB Intent Classifier",
            "method": "Complement Naive Bayes for short-text intent classification",
            "training_source": "data/training/intent_dataset.csv",
            "confidence_threshold": threshold,
        },
    }


def evaluate_intent_model() -> Dict[str, Any]:
    rows = _read_intent_dataset()
    if not rows:
        return {
            "dataset_path": str(INTENT_DATASET.relative_to(PROJECT_ROOT)),
            "dataset_size": 0,
            "classes": [],
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "full_report": {},
            "confusion_matrix": [],
        }

    texts = [row["text"] for row in rows]
    intents = [row["intent"] for row in rows]
    classes = sorted(set(intents))
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        intents,
        test_size=0.25,
        random_state=42,
        stratify=intents,
    )
    model = _intent_pipeline()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=classes)
    return {
        "dataset_path": str(INTENT_DATASET.relative_to(PROJECT_ROOT)),
        "dataset_size": len(rows),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "classes": classes,
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "confidence_threshold": INTENT_CONFIDENCE_THRESHOLD,
        "class_thresholds": INTENT_CLASS_THRESHOLDS,
        "full_report": report,
        "confusion_matrix": {
            "labels": classes,
            "matrix": matrix.tolist(),
        },
    }


def reload_intent_model() -> Dict[str, Any]:
    get_intent_model.cache_clear()
    return evaluate_intent_model()
