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
    kw_extractor = yake.KeywordExtractor(lan="id", n=1, top=3, features=None)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]


def find_relevant_diary(current_text: str, past_diaries: List[str]) -> Optional[str]:
    clean_diaries = [diary for diary in past_diaries if diary and diary.strip()]
    if not clean_diaries or not current_text:
        return None

    documents = clean_diaries + [current_text]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    current_vec = vectors[-1].reshape(1, -1)
    past_vecs = vectors[:-1]

    cosine_sim = cosine_similarity(current_vec, past_vecs).flatten()
    most_relevant_idx = cosine_sim.argsort()[-1]

    if cosine_sim[most_relevant_idx] > 0.1:
        return clean_diaries[most_relevant_idx]
    return None


def calculate_risk_level(
    text: str,
    screening_context: str = "",
    session_summary: str = "",
    client_risk: str = "",
) -> str:
    normalized_client_risk = (client_risk or "").strip().lower()
    if normalized_client_risk in {"high", "medium"}:
        return normalized_client_risk

    combined_text = f"{screening_context or ''} {session_summary or ''} {text or ''}".lower()

    score = 0
    for category, keywords in RISK_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined_text:
                score += RISK_WEIGHTS[category]

    crisis_pattern = re.compile(r"(bunuh diri|mengakhiri hidup|menyakiti diri|self harm)", re.IGNORECASE)
    severe_pattern = re.compile(r"(ekstrem|sangat berat|berat|severe|extremely severe)", re.IGNORECASE)
    medium_pattern = re.compile(r"(sedang|moderate)", re.IGNORECASE)

    if crisis_pattern.search(combined_text):
        return "high"
    if score >= 3 or severe_pattern.search(screening_context or ""):
        return "high"
    if score >= 2 or medium_pattern.search(screening_context or ""):
        return "medium"

    return "low"


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
    }
