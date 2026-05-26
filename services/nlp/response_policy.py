import re
from typing import Any, Dict, List, Optional


LENGTH_PROFILES: Dict[int, Dict[str, int]] = {
    1: {"min_words": 12, "max_words": 32, "max_tokens": 180},
    2: {"min_words": 70, "max_words": 170, "max_tokens": 480},
    3: {"min_words": 240, "max_words": 420, "max_tokens": 950},
    4: {"min_words": 360, "max_words": 760, "max_tokens": 2600},
}


def choose_desired_paragraphs(
    intent: str,
    intensity: str,
    word_count: int,
    short_listener_turn: bool,
) -> int:
    if short_listener_turn and intent not in {"safety_support", "advice_or_problem_solving"}:
        return 1
    if intent in {"meta_challenge", "clarification_followup"}:
        return 1
    if intent == "off_domain_redirect":
        return 1
    if intent == "casual_banter":
        return 1
    if intent == "casual_reference":
        return 2 if word_count >= 8 else 1
    if intent in {"response_feedback", "factual_or_product_question"}:
        return 2
    if intent == "check_in":
        return 1 if word_count <= 3 else 2
    if intent == "safety_support":
        return 2
    if intent in {"advice_or_problem_solving", "emotional_support"} or intensity in {"heavy", "tender"}:
        return 4
    return 2 if word_count <= 12 else 3


def target_words_for_paragraphs(desired_paragraphs: int) -> Dict[str, int]:
    profile = LENGTH_PROFILES.get(desired_paragraphs, LENGTH_PROFILES[2])
    return {
        "minimum": profile["min_words"],
        "maximum": profile["max_words"],
    }


def completion_token_budget(style_plan: Optional[Dict[str, Any]]) -> int:
    if not style_plan:
        return LENGTH_PROFILES[2]["max_tokens"]

    desired = int(style_plan.get("desired_paragraphs", 2) or 2)
    profile = LENGTH_PROFILES.get(desired, LENGTH_PROFILES[2])
    return profile["max_tokens"]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text or ""))


def cap_reply_length(reply: str, style_plan: Optional[Dict[str, Any]]) -> str:
    if not style_plan:
        return reply

    desired = int(style_plan.get("desired_paragraphs", 2) or 2)
    target_words = style_plan.get("target_words") or {}
    maximum_words = int(target_words.get("maximum", 0) or 0)
    if not maximum_words or desired >= 4 or word_count(reply) <= maximum_words:
        return reply

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", reply) if sentence.strip()]
    kept: List[str] = []
    for sentence in sentences:
        candidate = " ".join(kept + [sentence]).strip()
        if word_count(candidate) > maximum_words:
            break
        kept.append(sentence)

    if not kept and sentences:
        kept = [sentences[0]]

    return " ".join(kept).strip() or reply
