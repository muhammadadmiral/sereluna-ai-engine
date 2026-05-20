import re
from typing import Any, Dict, List

from services.nlp.lexicons import FILTER_TERM_ENTRIES


LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
        "@": "a",
        "$": "s",
        "!": "i",
    }
)
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def normalize_obfuscated_text(text: str) -> str:
    normalized = (text or "").lower().translate(LEET_TRANSLATION)
    normalized = re.sub(r"([a-z])\1{2,}", r"\1\1", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _levenshtein_at_most(left: str, right: str, max_distance: int) -> bool:
    if abs(len(left) - len(right)) > max_distance:
        return False
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = current[0]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            value = min(insertion, deletion, substitution)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return False
        previous = current
    return previous[-1] <= max_distance


def _match_term(term: str, normalized_text: str, compact_text: str, tokens: List[str]) -> Dict[str, Any] | None:
    normalized_term = normalize_obfuscated_text(term)
    compact_term = _compact(normalized_term)
    if not normalized_term:
        return None

    if normalized_term in normalized_text:
        return {"match_type": "normalized_exact", "normalized_term": normalized_term}
    if len(compact_term) >= 5 and compact_term in compact_text:
        return {"match_type": "compact_exact", "normalized_term": normalized_term}
    if " " not in normalized_term and len(normalized_term) >= 4:
        max_distance = 1 if len(normalized_term) < 7 else 2
        for token in tokens:
            if len(token) >= 4 and _levenshtein_at_most(token, normalized_term, max_distance):
                return {
                    "match_type": "fuzzy_token",
                    "normalized_term": normalized_term,
                    "matched_token": token,
                    "max_distance": max_distance,
                }
    return None


def analyze_preprocessing_filter(text: str) -> Dict[str, Any]:
    normalized_text = normalize_obfuscated_text(text)
    compact_text = _compact(normalized_text)
    tokens = normalized_text.split()
    matches: List[Dict[str, Any]] = []

    for row in FILTER_TERM_ENTRIES:
        term = row.get("term") or ""
        match = _match_term(term, normalized_text, compact_text, tokens)
        if not match:
            continue
        severity = row.get("severity") or "low"
        matches.append(
            {
                "term": term,
                "category": row.get("category") or "unknown",
                "severity": severity,
                **match,
            }
        )

    highest_severity = "none"
    if matches:
        highest_severity = max(matches, key=lambda item: SEVERITY_RANK.get(item["severity"], 0))["severity"]

    # Contextual check: if laughing or highly positive, toxicity might be slang intensifiers
    positive_cues = ["wkwk", "haha", "ngakak", "kocak", "lucu", "gokil", "keren", "mantap", "asik", "seru", "anjir", "anjrit"]
    has_positive_context = any(cue in normalized_text for cue in positive_cues)
    has_toxicity = any(item["category"] == "toxicity" for item in matches)
    
    # Suppress low/medium toxicity if positive context is present
    if has_positive_context and has_toxicity and highest_severity in {"low", "medium"}:
        has_toxicity = False

    return {
        "normalized_text": normalized_text,
        "compact_text": compact_text,
        "matches": matches,
        "has_crisis": any(item["category"] == "crisis" for item in matches),
        "has_toxicity": has_toxicity,
        "has_sensitive_content": any(item["category"] in {"sexual", "pii"} for item in matches),
        "highest_severity": highest_severity,
        "algorithm": {
            "name": "NLP Preprocessing Obfuscation Filter",
            "version": "1.0",
            "method": "leet normalization, punctuation stripping, compact matching, and bounded Levenshtein fuzzy token matching",
            "source": "data/lexicons/filter_terms.csv",
        },
    }
