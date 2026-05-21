import re
from typing import Any, Dict, List
from services.nlp.lexicons import (
    ADVICE_CUES, NEGATIVE_WORDS, ACHIEVEMENT_WORDS, QUESTION_CUES,
    CRISIS_PATTERNS, VIOLENCE_PATTERNS, SEXUAL_PATTERNS, PII_PATTERNS,
    RISK_WEIGHTS, RISK_THRESHOLDS, POSITIVE_WORDS
)
from services.nlp.utils import normalize_text, match_patterns, has_ambiguous_violence_context, is_greeting_only, contains_any

def assistant_turn_count(history_text: str) -> int:
    return len(re.findall(r"(?m)^Sereluna:", history_text or ""))

def relationship_stage(assistant_turns: int) -> str:
    if assistant_turns <= 0:
        return "new_room"
    if assistant_turns <= 2:
        return "warming_up"
    if assistant_turns <= 7:
        return "familiar"
    return "deep_room"

def detect_user_register(text: str, history_text: str) -> str:
    combined = normalize_text(f"{history_text or ''} {text or ''}")
    if re.search(r"\b(gua|gue|gw|lu|lo|elo)\b", combined):
        return "gue-lu santai"
    if re.search(r"\b(aku|kamu|dirimu)\b", combined):
        return "aku-kamu hangat"
    if re.search(r"\b(saya|anda)\b", combined):
        return "saya-anda lembut"
    return "aku-kamu santai"

def tone_guidance(stage: str, user_register: str) -> str:
    if stage == "new_room":
        return f"mulai hangat tapi tetap natural; pakai register {user_register}; jangan terlalu formal"
    if stage == "warming_up":
        return f"lanjutkan obrolan seperti sudah mulai kenal; pakai register {user_register}; kurangi sapaan pembuka"
    if stage == "familiar":
        return f"lebih santai dan responsif seperti teman yang sudah mengikuti cerita; pakai register {user_register}"
    return f"deep session: jangan reset konteks; pakai register {user_register}; respons harus terasa mengikuti alur chat panjang"

def continuity_guidance(stage: str) -> str:
    if stage == "new_room":
        return "boleh membuka ruang ngobrol, tapi tetap langsung nyambung ke pesan user."
    if stage == "warming_up":
        return "lanjutkan dari pesan sebelumnya dan jangan memperkenalkan Sereluna ulang."
    if stage == "familiar":
        return "rujuk detail obrolan sebelumnya kalau relevan, seolah Sereluna benar-benar mengikuti sesi ini."
    return "anggap ini room yang sudah panjang; jangan pakai pembuka generik, jangan recap berlebihan, dan respons sebagai lanjutan langsung dari konteks terakhir."

def classify_chat_intent(text: str, sentiment_score: int, risk_level: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return "empty"
    if risk_level == "high":
        return "safety_support"
    if is_greeting_only(normalized):
        return "check_in"
    if "?" in (text or "") or contains_any(normalized, ADVICE_CUES):
        return "advice_or_problem_solving"
    if sentiment_score <= 2 or contains_any(normalized, NEGATIVE_WORDS):
        return "emotional_support"
    if contains_any(normalized, ACHIEVEMENT_WORDS) or sentiment_score >= 4:
        return "celebration_or_progress"
    if contains_any(normalized, QUESTION_CUES):
        return "curious_question"
    return "reflective_companion"

def is_short_listener_turn(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False

    words = normalized.split()
    if len(words) > 9:
        return False

    if "?" in (text or ""):
        return False

    starter_cues = {
        "jadi", "terus", "abis", "habis", "trs", "trus", "lalu", "nah",
        "eh", "btw", "anjir", "jir", "wkwk", "wkwkwk", "haha", "hahaha",
    }
    casual_profanity = {"bangsat", "anjing", "kontol", "tai", "tolol", "goblok"}
    if any(cue in normalized for cue in starter_cues):
        return True
    if any(term in normalized for term in casual_profanity):
        return True

    return len(words) <= 4

def estimate_emotional_intensity(text: str, mood_signal: str, sentiment_score: int, risk_level: str) -> str:
    normalized = normalize_text(text)
    negative_hits = sum(1 for word in NEGATIVE_WORDS if word in normalized)
    if risk_level == "high":
        return "crisis"
    if risk_level == "medium" or negative_hits >= 3 or sentiment_score == 1:
        return "heavy"
    if negative_hits >= 1 or sentiment_score == 2 or (mood_signal or "").lower() in {"sad", "angry", "anxious"}:
        return "tender"
    if sentiment_score >= 4:
        return "light"
    return "neutral"

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
            "confidence": 1.0,
        }

    current_text = normalize_text(text)
    screening_text = normalize_text(screening_context)
    summary_text = normalize_text(session_summary)
    matches: List[Dict[str, Any]] = []
    score = 0

    crisis_current = match_patterns(current_text, CRISIS_PATTERNS)
    crisis_screening = match_patterns(screening_text, CRISIS_PATTERNS)
    crisis_summary = match_patterns(summary_text, CRISIS_PATTERNS)
    violence_current = match_patterns(current_text, VIOLENCE_PATTERNS)
    sexual_current = match_patterns(current_text, SEXUAL_PATTERNS)
    pii_current = match_patterns(current_text, PII_PATTERNS)

    if crisis_current:
        score += RISK_WEIGHTS["crisis"]
        matches.extend({"category": "crisis", "keyword": pattern, "weight": RISK_WEIGHTS["crisis"], "source": "current_text"} for pattern in crisis_current)
    if crisis_screening:
        matches.extend({"category": "crisis", "keyword": pattern, "weight": 0, "source": "screening_context"} for pattern in crisis_screening)
    if crisis_summary and not crisis_current:
        matches.extend({"category": "crisis", "keyword": pattern, "weight": 1, "source": "session_summary"} for pattern in crisis_summary)

    if violence_current:
        if has_ambiguous_violence_context(current_text):
            matches.extend({"category": "violence", "keyword": pattern, "weight": 0, "source": "ambiguous"} for pattern in violence_current)
        else:
            score += RISK_WEIGHTS["violence"]
            matches.extend({"category": "violence", "keyword": pattern, "weight": RISK_WEIGHTS["violence"], "source": "current_text"} for pattern in violence_current)

    if sexual_current:
        score += RISK_WEIGHTS["sexual"]
        matches.extend({"category": "sexual", "keyword": pattern, "weight": RISK_WEIGHTS["sexual"], "source": "current_text"} for pattern in sexual_current)

    if pii_current:
        score += RISK_WEIGHTS["pii"]
        matches.extend({"category": "pii", "keyword": pattern, "weight": RISK_WEIGHTS["pii"], "source": "current_text"} for pattern in pii_current)

    severe_pattern = re.compile(r"(ekstrem|sangat berat|berat|severe|extremely severe)", re.IGNORECASE)
    medium_pattern = re.compile(r"(sedang|moderate)", re.IGNORECASE)

    if crisis_current:
        level = "high"
        reason = "current_crisis_signal"
    elif score >= RISK_THRESHOLDS["high"]:
        level = "high"
        reason = "weighted_signal_score"
    elif severe_pattern.search(screening_context or ""):
        level = "medium"
        reason = "severe_screening_context"
    elif crisis_screening or crisis_summary:
        level = "medium"
        reason = "historical_crisis_context"
    elif score >= RISK_THRESHOLDS["medium"]:
        level = "medium"
        reason = "weighted_signal_score"
    elif medium_pattern.search(screening_context or ""):
        level = "medium"
        reason = "moderate_screening_context"
    else:
        level = "low"
        reason = "no_threshold_reached"

    confidence = 0.25
    if level == "high":
        confidence = 0.95 if (crisis_current or crisis_screening) else 0.8
    elif level == "medium":
        confidence = 0.6 if score >= 2 else 0.45

    return {
        "level": level,
        "score": score,
        "matches": matches,
        "reason": reason,
        "confidence": confidence,
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
    # Higher weight for laughter to override potential intensifiers
    laughter_cues = {"wkwk", "haha", "ngakak", "kocak", "lucu"}
    for word in POSITIVE_WORDS:
        if word in normalized_text:
            weight = 2 if any(cue in word for cue in laughter_cues) else 1
            lexical_score += weight
    for word in NEGATIVE_WORDS:
        if word in normalized_text:
            # If laughter is present, reduce impact of negative intensifiers like 'anjing' or 'bangsat'
            is_intensifier = word in {"anjing", "bangsat", "goblok", "tolol"}
            has_laughter = any(cue in normalized_text for cue in laughter_cues)
            if has_laughter and is_intensifier:
                continue
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
