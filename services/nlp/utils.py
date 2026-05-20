import re
from typing import List
import yake
from services.nlp.lexicons import GREETING_PATTERNS, AMBIGUOUS_VIOLENCE_PATTERNS

def extract_keywords(text: str) -> List[str]:
    if not text:
        return []
    try:
        kw_extractor = yake.KeywordExtractor(lan="id", n=1, top=3, features=None)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except Exception:
        return []

def normalize_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def match_patterns(text: str, patterns: List[str]) -> List[str]:
    matches: List[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(pattern)
    return matches

def has_ambiguous_violence_context(text: str) -> bool:
    return bool(match_patterns(text, AMBIGUOUS_VIOLENCE_PATTERNS))

def contains_any(text: str, cues: set[str]) -> bool:
    return any(cue in text for cue in cues)

def is_greeting_only(normalized_text: str) -> bool:
    return bool(match_patterns(normalized_text, GREETING_PATTERNS))
