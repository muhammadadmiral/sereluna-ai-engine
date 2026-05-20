import re
from typing import Any, Dict, List
from services.nlp.lexicons import (
    EMOTION_LEXICON, EMOTION_WEIGHTS, MOOD_TO_EMOTION,
    COGNITIVE_DISTORTION_PATTERNS, DISTORTION_LABELS, DISTORTION_REFRAME_TARGETS,
    DASS21_INDEXES, DASS21_THRESHOLDS
)
from services.nlp.utils import normalize_text

def _phrase_score(normalized_text: str, phrase: str) -> int:
    if phrase not in normalized_text:
        return 0
    return 2 if " " in phrase else 1

def build_emotion_profile(text: str, mood_signal: str, sentiment_score: int, risk_level: str) -> Dict[str, Any]:
    normalized = normalize_text(text)
    scores: Dict[str, int] = {emotion: 0 for emotion in EMOTION_LEXICON}
    evidence: Dict[str, List[str]] = {emotion: [] for emotion in EMOTION_LEXICON}

    for emotion, words in EMOTION_LEXICON.items():
        for word in words:
            score = _phrase_score(normalized, word)
            if score:
                score *= EMOTION_WEIGHTS.get(emotion, {}).get(word, 1)
                scores[emotion] += score
                evidence[emotion].append(word)

    mood_emotion = MOOD_TO_EMOTION.get((mood_signal or "").strip().lower())
    if mood_emotion:
        scores[mood_emotion] += 2
        evidence[mood_emotion].append(f"mood_signal:{mood_signal}")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary_emotion, primary_score = ranked[0] if ranked else ("neutral", 0)
    if primary_score == 0:
        primary_emotion = "distress" if sentiment_score <= 2 else "neutral"

    raw_intensity = primary_score
    if sentiment_score <= 2:
        raw_intensity += 1
    if risk_level == "medium":
        raw_intensity += 2
    if risk_level == "high":
        raw_intensity += 4

    if raw_intensity >= 6:
        intensity = "high"
    elif raw_intensity >= 3:
        intensity = "medium"
    elif raw_intensity >= 1:
        intensity = "low"
    else:
        intensity = "neutral"

    secondary = [
        {"emotion": emotion, "score": score, "evidence": evidence[emotion][:4]}
        for emotion, score in ranked[1:4]
        if score > 0
    ]

    return {
        "primary_emotion": primary_emotion,
        "intensity": intensity,
        "scores": scores,
        "evidence": {emotion: hits for emotion, hits in evidence.items() if hits},
        "secondary_emotions": secondary,
        "algorithm": {
            "name": "Sereluna Emotion Lexicon Profiler",
            "version": "1.0",
            "method": "weighted Indonesian emotion lexicon plus client mood signal",
        },
    }

def detect_cognitive_distortions(text: str) -> Dict[str, Any]:
    normalized = normalize_text(text)
    distortions: List[Dict[str, Any]] = []

    for distortion, patterns in COGNITIVE_DISTORTION_PATTERNS.items():
        matched_patterns = [pattern for pattern in patterns if re.search(pattern, normalized, flags=re.IGNORECASE)]
        if matched_patterns:
            distortions.append(
                {
                    "type": distortion,
                    "label": DISTORTION_LABELS.get(distortion, distortion),
                    "evidence_patterns": matched_patterns,
                }
            )

    distortion_types = {item["type"] for item in distortions}
    reframe_targets = []
    for distortion_type in distortion_types:
        target = DISTORTION_REFRAME_TARGETS.get(distortion_type)
        if target and target not in reframe_targets:
            reframe_targets.append(target)

    return {
        "detected": distortions,
        "count": len(distortions),
        "reframe_targets": reframe_targets,
        "algorithm": {
            "name": "CBT-Inspired Cognitive Distortion Pattern Miner",
            "version": "1.0",
            "method": "regular-expression pattern matching over normalized Indonesian chat text",
        },
    }

def _severity_for(category: str, score: int) -> str:
    for threshold, severity in DASS21_THRESHOLDS[category]:
        if score <= threshold:
            return severity
    return "extremely severe"

def score_dass21(answers: List[int]) -> Dict[str, Any]:
    if len(answers) != 21:
        raise ValueError("DASS-21 requires exactly 21 answers")
    if any(answer < 0 or answer > 3 for answer in answers):
        raise ValueError("DASS-21 answers must be in range 0..3")

    scores: Dict[str, int] = {}
    severity: Dict[str, str] = {}
    for category, indexes in DASS21_INDEXES.items():
        category_score = sum(answers[index] for index in indexes) * 2
        scores[category] = category_score
        severity[category] = _severity_for(category, category_score)

    summary = (
        "DASS-21: "
        f"Depresi {scores['depression']} ({severity['depression']}), "
        f"Anxiety {scores['anxiety']} ({severity['anxiety']}), "
        f"Stress {scores['stress']} ({severity['stress']})."
    )

    return {
        "scores": scores,
        "severity": severity,
        "summary": summary,
        "algorithm": {
            "name": "DASS-21 scoring",
            "description": "Sum selected item groups and multiply by 2, then map scores to severity bands.",
            "item_indexes": DASS21_INDEXES,
            "thresholds": {
                category: [
                    {"max_score": threshold, "severity": severity_label}
                    for threshold, severity_label in thresholds
                ]
                for category, thresholds in DASS21_THRESHOLDS.items()
            },
        },
    }

def predict_implicit_dass21(
    emotion_profile: Dict[str, Any],
    distortion_profile: Dict[str, Any],
    sentiment_score: int,
) -> Dict[str, Any]:
    """
    Predict implicit DASS-21 signals based on current session NLP metrics.
    This is used for 'Proactive Mental Health Support' in the thesis.
    """
    primary_emotion = emotion_profile.get("primary_emotion", "neutral")
    intensity = emotion_profile.get("intensity", "neutral")
    distortion_count = distortion_profile.get("count", 0)

    # Base scores
    d_score, a_score, s_score = 0, 0, 0

    # Emotion mapping to DASS categories
    if primary_emotion == "sadness":
        d_score += 2 if intensity == "low" else 4 if intensity == "medium" else 6
    elif primary_emotion == "anxiety":
        a_score += 2 if intensity == "low" else 4 if intensity == "medium" else 6
    elif primary_emotion in {"anger", "fatigue"}:
        s_score += 2 if intensity == "low" else 4 if intensity == "medium" else 6
    elif primary_emotion == "shame":
        d_score += 3

    # Distortion impact
    if distortion_count > 0:
        s_score += distortion_count * 2
        d_score += distortion_count

    # Sentiment impact
    if sentiment_score <= 2:
        d_score += 2
        a_score += 2

    return {
        "predicted_scores": {"depression": d_score, "anxiety": a_score, "stress": s_score},
        "signals": {
            "depression": d_score >= 6,
            "anxiety": a_score >= 6,
            "stress": s_score >= 8,
        },
        "method": "Implicit NLP mapping to DASS-21 categories",
    }
