import csv
from pathlib import Path
from typing import Dict, List


LEXICON_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "lexicons"


def _read_csv(filename: str) -> List[Dict[str, str]]:
    path = LEXICON_DIR / filename
    with path.open("r", encoding="utf-8", newline="") as file:
        rows: List[Dict[str, str]] = []
        for row in csv.DictReader(file):
            cleaned: Dict[str, str] = {}
            for key, value in row.items():
                if key is None:
                    continue
                if isinstance(value, list):
                    cleaned[key] = ",".join(value).strip()
                else:
                    cleaned[key] = (value or "").strip()
            rows.append(cleaned)
        return rows


def _int(value: str, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


SENTIMENT_LEXICON_ENTRIES = _read_csv("sentiment_lexicon.csv")
EMOTION_LEXICON_ENTRIES = _read_csv("emotion_lexicon.csv")
RISK_PATTERN_ENTRIES = _read_csv("risk_patterns.csv")
COGNITIVE_DISTORTION_ENTRIES = _read_csv("cognitive_distortion_patterns.csv")
CHAT_CUE_ENTRIES = _read_csv("chat_cues.csv")
MOOD_EMOTION_ENTRIES = _read_csv("mood_emotion_map.csv")
FILTER_TERM_ENTRIES = _read_csv("filter_terms.csv")

NEGATIVE_WORDS = {
    row["term"]
    for row in SENTIMENT_LEXICON_ENTRIES
    if row.get("polarity") == "negative" and row.get("term")
}
POSITIVE_WORDS = {
    row["term"]
    for row in SENTIMENT_LEXICON_ENTRIES
    if row.get("polarity") == "positive" and row.get("term")
}

EMOTION_LEXICON: Dict[str, List[str]] = {}
EMOTION_WEIGHTS: Dict[str, Dict[str, int]] = {}
for row in EMOTION_LEXICON_ENTRIES:
    emotion = row.get("emotion")
    term = row.get("term")
    if not emotion or not term:
        continue
    EMOTION_LEXICON.setdefault(emotion, []).append(term)
    EMOTION_WEIGHTS.setdefault(emotion, {})[term] = _int(row.get("weight", "1"))

MOOD_TO_EMOTION = {
    row["mood"]: row["emotion"]
    for row in MOOD_EMOTION_ENTRIES
    if row.get("mood") and row.get("emotion")
}

RISK_WEIGHTS = {"crisis": 3, "violence": 2, "sexual": 1, "pii": 1}
RISK_THRESHOLDS = {"medium": 2, "high": 3}

CRISIS_PATTERNS = [
    row["pattern"]
    for row in RISK_PATTERN_ENTRIES
    if row.get("category") == "crisis" and row.get("kind") == "current"
]
VIOLENCE_PATTERNS = [
    row["pattern"]
    for row in RISK_PATTERN_ENTRIES
    if row.get("category") == "violence" and row.get("kind") == "current"
]
AMBIGUOUS_VIOLENCE_PATTERNS = [
    row["pattern"]
    for row in RISK_PATTERN_ENTRIES
    if row.get("category") == "violence" and row.get("kind") == "ambiguous"
]
SEXUAL_PATTERNS = [
    row["pattern"]
    for row in RISK_PATTERN_ENTRIES
    if row.get("category") == "sexual" and row.get("kind") == "current"
]
PII_PATTERNS = [
    row["pattern"]
    for row in RISK_PATTERN_ENTRIES
    if row.get("category") == "pii" and row.get("kind") == "current"
]

COGNITIVE_DISTORTION_PATTERNS: Dict[str, List[str]] = {}
DISTORTION_LABELS: Dict[str, str] = {}
DISTORTION_REFRAME_TARGETS: Dict[str, str] = {}
for row in COGNITIVE_DISTORTION_ENTRIES:
    distortion_type = row.get("type")
    pattern = row.get("pattern")
    if not distortion_type or not pattern:
        continue
    COGNITIVE_DISTORTION_PATTERNS.setdefault(distortion_type, []).append(pattern)
    DISTORTION_LABELS[distortion_type] = row.get("label") or distortion_type
    if row.get("reframe_target"):
        DISTORTION_REFRAME_TARGETS[distortion_type] = row["reframe_target"]

QUESTION_CUES = {
    row["term"]
    for row in CHAT_CUE_ENTRIES
    if row.get("type") == "question" and row.get("term")
}
ADVICE_CUES = {
    row["term"]
    for row in CHAT_CUE_ENTRIES
    if row.get("type") == "advice" and row.get("term")
}
ACHIEVEMENT_WORDS = {
    row["term"]
    for row in CHAT_CUE_ENTRIES
    if row.get("type") == "achievement" and row.get("term")
}
GREETING_PATTERNS = [
    row["term"]
    for row in CHAT_CUE_ENTRIES
    if row.get("type") == "greeting_pattern" and row.get("term")
]
BANNED_CHAT_OPENERS = [
    row["term"]
    for row in CHAT_CUE_ENTRIES
    if row.get("type") == "banned_opener" and row.get("term")
]

EMOJI_ROTATION = [
    "\U0001f331",
    "\U0001f319",
    "\U0001f642",
    "\U0001f90d",
    "\u2728",
]

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
