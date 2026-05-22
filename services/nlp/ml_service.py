import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import normalize

from services.nlp.lexicons import EMOTION_LEXICON, EMOTION_LEXICON_ENTRIES, EMOTION_WEIGHTS


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUPERVISED_EMOTION_DATASET = PROJECT_ROOT / "data" / "training" / "emotion_dataset.csv"
SUPERVISED_CONFIDENCE_THRESHOLD = 0.35


def _normalize_text(text: str) -> str:
    import re

    normalized = (text or "").lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _phrase_score(normalized_text: str, phrase: str) -> int:
    if phrase not in normalized_text:
        return 0
    return 2 if " " in phrase else 1


@lru_cache(maxsize=1)
def _emotion_centroid_model() -> Dict[str, Any]:
    training_rows = [
        {
            "term": row["term"],
            "emotion": row["emotion"],
            "weight": int(row.get("weight") or 1),
        }
        for row in EMOTION_LEXICON_ENTRIES
        if row.get("term") and row.get("emotion")
    ]
    terms = [row["term"] for row in training_rows]
    labels = [row["emotion"] for row in training_rows]
    weights = np.array([max(1, row["weight"]) for row in training_rows], dtype=float)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True)
    vectors = normalize(vectorizer.fit_transform(terms))

    centroids: Dict[str, Any] = {}
    for emotion in sorted(set(labels)):
        indexes = [index for index, label in enumerate(labels) if label == emotion]
        if not indexes:
            continue
        class_weights = weights[indexes]
        class_vectors = vectors[indexes].multiply(class_weights[:, None])
        centroid = sparse.csr_matrix(class_vectors.sum(axis=0)) / class_weights.sum()
        centroids[emotion] = normalize(centroid)

    return {
        "vectorizer": vectorizer,
        "centroids": centroids,
        "training_rows": len(training_rows),
        "classes": sorted(centroids),
    }


def classify_emotion_ml(text: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    if not normalized:
        return {
            "predicted_emotion": "neutral",
            "confidence": 0.0,
            "scores": {},
            "algorithm": {
                "name": "TF-IDF Nearest-Centroid Emotion Classifier",
                "version": "1.0",
                "training_source": "data/lexicons/emotion_lexicon.csv",
            },
        }

    model = _emotion_centroid_model()
    query_vector = normalize(model["vectorizer"].transform([normalized]))
    scores = {
        emotion: float(cosine_similarity(query_vector, centroid)[0][0])
        for emotion, centroid in model["centroids"].items()
    }
    predicted_emotion, confidence = max(scores.items(), key=lambda item: item[1])
    if confidence < 0.02:
        predicted_emotion = "neutral"

    return {
        "predicted_emotion": predicted_emotion,
        "confidence": round(float(confidence), 4),
        "scores": {emotion: round(score, 4) for emotion, score in sorted(scores.items())},
        "algorithm": {
            "name": "TF-IDF Nearest-Centroid Emotion Classifier",
            "version": "1.0",
            "method": "fit TF-IDF character n-gram vectors from curated emotion lexicon, then classify by cosine distance to class centroids",
            "training_source": "data/lexicons/emotion_lexicon.csv",
            "training_rows": model["training_rows"],
            "classes": model["classes"],
        },
    }


def _read_supervised_emotion_dataset() -> List[Dict[str, str]]:
    with SUPERVISED_EMOTION_DATASET.open("r", encoding="utf-8", newline="") as file:
        rows = [
            {"text": (row.get("text") or "").strip(), "label": (row.get("label") or "").strip()}
            for row in csv.DictReader(file)
        ]
    return [row for row in rows if row["text"] and row["label"]]


from services.nlp.lexicons import (
    EMOTION_LEXICON, EMOTION_LEXICON_ENTRIES, EMOTION_WEIGHTS,
    POSITIVE_WORDS, NEGATIVE_WORDS, COGNITIVE_DISTORTION_PATTERNS,
    CRISIS_PATTERNS
)

class MentalHealthFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Hybrid feature extractor that mines signals from:
    1. Emotion Lexicon (Curated keywords)
    2. Sentiment Polarity (Positive vs Negative density)
    3. Cognitive Distortions (Pattern matching)
    4. Crisis/Risk Signals
    """
    def __init__(self) -> None:
        self.emotions = sorted(EMOTION_LEXICON)
        self.distortion_types = sorted(COGNITIVE_DISTORTION_PATTERNS.keys())

    def fit(self, texts, y=None):
        return self

    def _get_lexicon_features(self, normalized_text: str) -> List[float]:
        feats = []
        for emotion in self.emotions:
            score = 0.0
            for term in EMOTION_LEXICON.get(emotion, []):
                if term in normalized_text:
                    # Multiplier for multi-word phrases
                    score += (2.0 if " " in term else 1.0) * EMOTION_WEIGHTS.get(emotion, {}).get(term, 1)
            feats.append(score)
        total = sum(feats)
        return [f / total for f in feats] if total > 0 else [0.0] * len(feats)

    def transform(self, texts):
        all_features = []
        for text in texts:
            norm = _normalize_text(str(text))
            words = set(norm.split())
            
            # 1. Emotion Lexicon Density
            emo_feats = self._get_lexicon_features(norm)
            
            # 2. Sentiment Polarity Density
            pos_count = len(words.intersection(POSITIVE_WORDS))
            neg_count = len(words.intersection(NEGATIVE_WORDS))
            total_words = len(words) or 1
            sentiment_feats = [pos_count / total_words, neg_count / total_words]
            
            # 3. Cognitive Distortion Flags
            distortion_feats = [
                1.0 if any(p in norm for p in COGNITIVE_DISTORTION_PATTERNS[dt]) else 0.0
                for dt in self.distortion_types
            ]
            
            # 4. Risk Density
            risk_count = sum(1 for p in CRISIS_PATTERNS if p in norm)
            risk_feats = [risk_count / total_words]
            
            # Combine all engineered features
            all_features.append(emo_feats + sentiment_feats + distortion_feats + risk_feats)
            
        return sparse.csr_matrix(np.asarray(all_features, dtype=float))

def _emotion_classification_pipeline() -> Pipeline:
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
                                ngram_range=(1, 3), # Increased to 3 for more context
                                lowercase=True,
                                sublinear_tf=True,
                                max_df=0.85,
                                min_df=2
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 6), # Increased to 6
                                lowercase=True,
                                sublinear_tf=True,
                                max_df=0.85,
                                min_df=3
                            ),
                        ),
                        ("mh_expert_features", MentalHealthFeatureExtractor()),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=1.5,
                    random_state=42,
                    solver='lbfgs' # Default solver, supports multinomial multiclass
                ),
            ),
        ]
    )


from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

# ... rest of imports ...

# --- Global Model Cache (Singleton for Skripsi Demo) ---
_GLOBAL_MODEL_BUNDLE = None

def get_trained_model():
    """
    Ensures the model is trained only once and stored in memory.
    This satisfies the 'Data Mining' requirement without slowing down every request.
    """
    global _GLOBAL_MODEL_BUNDLE
    if _GLOBAL_MODEL_BUNDLE is None:
        print("🛠️ [SKRIPS] Initializing Data Mining & ML Training Pipeline...")
        _GLOBAL_MODEL_BUNDLE = _supervised_emotion_model_full_process()
    return _GLOBAL_MODEL_BUNDLE

def _supervised_emotion_model_full_process() -> Dict[str, Any]:
    """
    The full academic pipeline: Data loading, Feature Extraction, and Training.
    """
    rows = _read_supervised_emotion_dataset()
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    classes = sorted(set(labels))

    # Cross-validation for academic rigor (sidang)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = _emotion_classification_pipeline()
    
    # We run this once at startup so the metrics in the logs are real
    cv_results = cross_validate(
        pipeline, texts, labels, 
        cv=skf, 
        scoring=['accuracy', 'f1_macro'],
        return_train_score=False
    )

    mean_accuracy = float(np.mean(cv_results['test_accuracy']))
    std_accuracy = float(np.std(cv_results['test_accuracy']))
    macro_f1 = float(np.mean(cv_results['test_f1_macro']))

    # Final production fit
    production_model = _emotion_classification_pipeline()
    production_model.fit(texts, labels)

    evaluation = {
        "dataset_path": str(SUPERVISED_EMOTION_DATASET.relative_to(PROJECT_ROOT)),
        "dataset_size": len(rows),
        "train_size": len(rows),
        "classes": classes,
        "accuracy": round(mean_accuracy, 4),
        "accuracy_std": round(std_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "confidence_threshold": SUPERVISED_CONFIDENCE_THRESHOLD,
        "cv_folds": 5
    }

    return {
        "model": production_model,
        "evaluation": evaluation,
    }

def evaluate_supervised_emotion_model() -> Dict[str, Any]:
    """
    Returns the evaluation metrics for the professor's demo.
    """
    return get_trained_model()["evaluation"]

def reload_supervised_emotion_model() -> Dict[str, Any]:
    """
    Clears the cache and re-trains the model.
    """
    global _GLOBAL_MODEL_BUNDLE
    _GLOBAL_MODEL_BUNDLE = None
    return evaluate_supervised_emotion_model()

def classify_emotion_supervised(text: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    if not normalized:
        return {"predicted_emotion": "neutral", "confidence": 0.0, "accepted": False}

    # Use the cached global model
    model_bundle = get_trained_model()
    model = model_bundle["model"]
    
    probabilities = model.predict_proba([normalized])[0]
    classes = list(model.classes_)
    ranked = sorted(
        (
            {"emotion": emotion, "probability": round(float(probability), 4)}
            for emotion, probability in zip(classes, probabilities)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    top = ranked[0]
    confidence = float(top["probability"])

    return {
        "predicted_emotion": top["emotion"],
        "confidence": round(confidence, 4),
        "accepted": confidence >= SUPERVISED_CONFIDENCE_THRESHOLD,
        "top_probabilities": ranked[:3],
        "evaluation_summary": model_bundle["evaluation"],
        "algorithm": {
            "name": "TF-IDF Logistic Regression Emotion Classifier",
            "method": "Hybrid TF-IDF + Mental Health Expert Features",
            "training_source": "data/training/emotion_dataset.csv"
        },
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_supervised_emotion_model(), indent=2, ensure_ascii=False))
