import re
from typing import Any, Dict, List, Optional

import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RISK_KEYWORDS = {
    "crisis": ["bunuh diri", "mengakhiri hidup", "self harm", "menyakiti diri", "mati saja"],
    "violence": ["bunuh", "pukul", "bacok", "tusuk", "ledak"],
    "sexual": ["seks", "porno", "mesum"],
    "pii": ["nik", "ktp", "alamat lengkap", "nomor kartu"],
}
RISK_WEIGHTS = {"crisis": 3, "violence": 2, "sexual": 1, "pii": 1}
RISK_THRESHOLDS = {"medium": 2, "high": 3}
DIARY_RETRIEVAL_THRESHOLD = 0.1

NEGATIVE_WORDS = {
    "sedih", "capek", "lelah", "takut", "cemas", "khawatir", "panik",
    "hancur", "gagal", "sendiri", "kesepian", "marah", "stress", "stres",
    "tertekan", "putus asa", "menangis", "buruk", "sakit",
}
POSITIVE_WORDS = {
    "senang", "lega", "tenang", "bahagia", "semangat", "baik", "aman",
    "bersyukur", "nyaman", "membaik", "kuat", "terbantu",
}

DASS21_INDEXES = {
    "depression": [2, 4, 9, 12, 15, 16, 20],
    "anxiety": [1, 3, 6, 8, 14, 18, 19],
    "stress": [0, 5, 7, 10, 11, 13, 17],
}

DASS21_THRESHOLDS = {
    "depression": [(9, "normal"), (13, "mild"), (20, "moderate"), (27, "severe")],
    "anxiety": [(7, "normal"), (9, "mild"), (14, "moderate"), (19, "severe")],
    "stress": [(14, "normal"), (18, "mild"), (25, "moderate"), (33, "severe")],
}


def extract_keywords(text: str) -> List[str]:
    if not text:
        return []
    try:
        kw_extractor = yake.KeywordExtractor(lan="id", n=1, top=3, features=None)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except Exception:
        return []


def find_relevant_diary_with_score(
    current_text: str,
    past_diaries: List[str],
    threshold: float = DIARY_RETRIEVAL_THRESHOLD,
) -> Dict[str, Any]:
    clean_diaries = [diary for diary in past_diaries if diary and diary.strip()]
    if not clean_diaries or not current_text:
        return {
            "diary": None,
            "similarity": 0.0,
            "index": None,
            "threshold": threshold,
        }

    documents = clean_diaries + [current_text]
    try:
        vectorizer = TfidfVectorizer().fit_transform(documents)
        vectors = vectorizer.toarray()
    except ValueError:
        return {
            "diary": None,
            "similarity": 0.0,
            "index": None,
            "threshold": threshold,
        }

    current_vec = vectors[-1].reshape(1, -1)
    past_vecs = vectors[:-1]

    cosine_sim = cosine_similarity(current_vec, past_vecs).flatten()
    most_relevant_idx = cosine_sim.argsort()[-1]
    similarity = float(cosine_sim[most_relevant_idx])

    return {
        "diary": clean_diaries[most_relevant_idx] if similarity > threshold else None,
        "similarity": similarity,
        "index": int(most_relevant_idx),
        "threshold": threshold,
    }


def find_relevant_diary(current_text: str, past_diaries: List[str]) -> Optional[str]:
    return find_relevant_diary_with_score(current_text, past_diaries)["diary"]


def classify_risk(
    text: str,
    screening_context: str = "",
    session_summary: str = "",
    client_risk: str = "",
) -> Dict[str, Any]:
    normalized_client_risk = (client_risk or "").strip().lower()
    if normalized_client_risk in {"high", "medium"}:
        return {
            "level": normalized_client_risk,
            "score": RISK_THRESHOLDS[normalized_client_risk],
            "matches": [],
            "reason": "client_override",
        }

    combined_text = f"{screening_context or ''} {session_summary or ''} {text or ''}".lower()
    matches: List[Dict[str, Any]] = []
    score = 0

    for category, keywords in RISK_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined_text:
                weight = RISK_WEIGHTS[category]
                score += weight
                matches.append({"category": category, "keyword": keyword, "weight": weight})

    crisis_pattern = re.compile(r"(bunuh diri|mengakhiri hidup|menyakiti diri|self harm)", re.IGNORECASE)
    severe_pattern = re.compile(r"(ekstrem|sangat berat|berat|severe|extremely severe)", re.IGNORECASE)
    medium_pattern = re.compile(r"(sedang|moderate)", re.IGNORECASE)

    if crisis_pattern.search(combined_text):
        level = "high"
        reason = "crisis_pattern"
    elif score >= RISK_THRESHOLDS["high"]:
        level = "high"
        reason = "weighted_keyword_score"
    elif severe_pattern.search(screening_context or ""):
        level = "high"
        reason = "severe_screening_context"
    elif score >= RISK_THRESHOLDS["medium"]:
        level = "medium"
        reason = "weighted_keyword_score"
    elif medium_pattern.search(screening_context or ""):
        level = "medium"
        reason = "moderate_screening_context"
    else:
        level = "low"
        reason = "no_threshold_reached"

    return {
        "level": level,
        "score": score,
        "matches": matches,
        "reason": reason,
        "thresholds": RISK_THRESHOLDS,
    }


def calculate_risk_level(
    text: str,
    screening_context: str = "",
    session_summary: str = "",
    client_risk: str = "",
) -> str:
    return classify_risk(text, screening_context, session_summary, client_risk)["level"]


def calculate_sentiment_score(text: str, mood_signal: str = "") -> int:
    normalized_text = (text or "").lower()
    normalized_mood = (mood_signal or "").lower()

    lexical_score = 0
    for word in POSITIVE_WORDS:
        if word in normalized_text:
            lexical_score += 1
    for word in NEGATIVE_WORDS:
        if word in normalized_text:
            lexical_score -= 1

    if normalized_mood in {"negative", "sad", "angry", "anxious", "stress", "stressed"}:
        lexical_score -= 1
    if normalized_mood in {"positive", "happy", "calm", "good"}:
        lexical_score += 1

    if lexical_score <= -3:
        return 1
    if lexical_score <= -1:
        return 2
    if lexical_score == 0:
        return 3
    if lexical_score <= 2:
        return 4
    return 5


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


def build_context_algorithm_result(
    text: str,
    mood_signal: str,
    screening_context: str,
    session_summary: str,
    past_diaries: List[str],
) -> Dict[str, Any]:
    risk = classify_risk(
        text=text,
        screening_context=screening_context,
        session_summary=session_summary,
    )
    retrieval = find_relevant_diary_with_score(text, past_diaries)
    keywords = extract_keywords(text)
    sentiment_score = calculate_sentiment_score(text, mood_signal)

    return {
        "risk_level": risk["level"],
        "risk": risk,
        "sentiment_score": sentiment_score,
        "keywords": keywords,
        "relevant_diary": retrieval["diary"],
        "retrieval": retrieval,
        "algorithms": {
            "main": [
                "weighted_rule_based_risk_classification",
                "tfidf_cosine_similarity_diary_retrieval",
            ],
            "supporting": ["lexicon_based_sentiment_scoring"],
        },
    }
