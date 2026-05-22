from typing import Any, Dict, List
from services.nlp.ml_service import (
    SUPERVISED_CONFIDENCE_THRESHOLD,
    classify_emotion_ml,
    classify_emotion_supervised,
    evaluate_supervised_emotion_model,
)
from services.nlp.preprocessing_filter import analyze_preprocessing_filter
from services.nlp.lexicons import RISK_THRESHOLDS
from services.nlp.utils import normalize_text, extract_keywords, is_greeting_only
from services.nlp.session import (
    classify_risk, calculate_sentiment_score
)
from services.nlp.clinical import (
    build_emotion_profile, detect_cognitive_distortions, predict_implicit_dass21
)
from services.nlp.logic import (
    select_coping_pathway, find_relevant_diary_with_score, DIARY_RETRIEVAL_THRESHOLD
)

def build_context_algorithm_result(
    text: str,
    mood_signal: str,
    screening_context: str,
    session_summary: str,
    past_diaries: List[str],
) -> Dict[str, Any]:
    preprocessing_filter = analyze_preprocessing_filter(text)
    # Use normalized text as the primary source for further analysis to avoid doubling
    analysis_text = preprocessing_filter.get("normalized_text", text)
    
    current_is_greeting = is_greeting_only(normalize_text(text))
    risk = classify_risk(
        text=analysis_text,
        screening_context="" if current_is_greeting else screening_context,
        session_summary="" if current_is_greeting else session_summary,
    )
    if preprocessing_filter.get("has_crisis") and risk["level"] != "high":
        risk = {
            **risk,
            "level": "high",
            "reason": "preprocessing_crisis_filter",
            "confidence": 0.92,
            "matches": risk.get("matches", []) + [
                {
                    "category": "crisis",
                    "keyword": item["term"],
                    "weight": RISK_THRESHOLDS["high"],
                    "source": item["match_type"],
                }
                for item in preprocessing_filter.get("matches", [])
                if item.get("category") == "crisis"
            ],
        }
    retrieval = (
        {"diary": None, "similarity": 0.0, "index": None, "threshold": DIARY_RETRIEVAL_THRESHOLD}
        if current_is_greeting
        else find_relevant_diary_with_score(analysis_text, past_diaries)
    )
    keywords = extract_keywords(analysis_text)
    sentiment_score = calculate_sentiment_score(analysis_text, mood_signal)
    emotion_profile = build_emotion_profile(analysis_text, mood_signal, sentiment_score, risk["level"])
    
    ml_emotion = (
        {
            "predicted_emotion": "neutral",
            "confidence": 0.0,
            "scores": {},
            "algorithm": {
                "name": "TF-IDF Nearest-Centroid Emotion Classifier",
                "version": "1.0",
                "skipped": "neutral_greeting_current_room_only",
                "training_source": "data/lexicons/emotion_lexicon.csv",
            },
        }
        if current_is_greeting
        else classify_emotion_ml(analysis_text)
    )
    supervised_emotion = (
        {
            "predicted_emotion": "neutral",
            "confidence": 0.0,
            "accepted": False,
            "top_probabilities": [],
            "algorithm": {
                "name": "TF-IDF Logistic Regression Emotion Classifier",
                "version": "1.0",
                "skipped": "neutral_greeting_current_room_only",
                "training_source": "data/training/emotion_dataset.csv",
                "confidence_threshold": SUPERVISED_CONFIDENCE_THRESHOLD,
            },
        }
        if current_is_greeting
        else classify_emotion_supervised(analysis_text)
    )
    
    emotion_profile["ml_prediction"] = ml_emotion
    emotion_profile["supervised_prediction"] = supervised_emotion
    
    if (
        emotion_profile["primary_emotion"] in {"neutral", "distress"}
        and supervised_emotion["predicted_emotion"] != "neutral"
        and supervised_emotion.get("accepted")
    ):
        emotion_profile["primary_emotion"] = supervised_emotion["predicted_emotion"]
        emotion_profile["intensity"] = "low"
    elif emotion_profile["primary_emotion"] in {"neutral", "distress"} and ml_emotion["predicted_emotion"] != "neutral":
        emotion_profile["primary_emotion"] = ml_emotion["predicted_emotion"]
        emotion_profile["intensity"] = "low"
        
    distortion_profile = detect_cognitive_distortions(analysis_text)
    implicit_dass21 = predict_implicit_dass21(emotion_profile, distortion_profile, sentiment_score)
    coping_pathway = select_coping_pathway(
        text=text,
        risk_level=risk["level"],
        sentiment_score=sentiment_score,
        emotion_profile=emotion_profile,
        distortion_profile=distortion_profile,
    )

    return {
        "risk_level": risk["level"],
        "preprocessing_filter": preprocessing_filter,
        "risk": risk,
        "sentiment_score": sentiment_score,
        "keywords": keywords,
        "relevant_diary": retrieval["diary"],
        "retrieval": retrieval,
        "emotion_profile": emotion_profile,
        "implicit_dass21": implicit_dass21,
        "ml_emotion_classifier": ml_emotion,
        "supervised_emotion_classifier": supervised_emotion,
        "supervised_model_evaluation": evaluate_supervised_emotion_model(),
        "cognitive_distortions": distortion_profile,
        "coping_pathway": coping_pathway,
        "algorithms": {
            "main": [
                "weighted_rule_based_risk_classification",
                "tfidf_cosine_similarity_diary_retrieval",
                "emotion_lexicon_intensity_profile",
                "tfidf_nearest_centroid_emotion_classifier",
                "tfidf_logistic_regression_emotion_classifier",
                "nlp_preprocessing_obfuscation_filter",
                "cognitive_distortion_pattern_mining",
                "coping_pathway_decision_tree",
                "implicit_dass21_proactive_screening",
            ],
            "supporting": ["lexicon_based_sentiment_scoring", "yake_keyword_extraction"],
        },
    }
