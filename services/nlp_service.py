import re
from typing import Any, Dict, List, Optional

import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RISK_KEYWORDS = {
    "crisis": ["bunuh diri", "mengakhiri hidup", "self harm", "menyakiti diri", "mati saja"],
    "violence": ["bunuh", "bacok", "tusuk", "ledak", "hajar", "serang"],
    "sexual": ["seks", "porno", "mesum"],
    "pii": ["nik", "ktp", "alamat lengkap", "nomor kartu"],
}
RISK_WEIGHTS = {"crisis": 3, "violence": 2, "sexual": 1, "pii": 1}
RISK_THRESHOLDS = {"medium": 2, "high": 3}
DIARY_RETRIEVAL_THRESHOLD = 0.1

CRISIS_PATTERNS = [
    r"\bbunuh\s*diri\b",
    r"\bmengakhiri\s+hidup\b",
    r"\bmenyakiti\s+diri\b",
    r"\bself\s*harm\b",
    r"\bmati\s+saja\b",
]
VIOLENCE_PATTERNS = [
    r"\bbacok\b",
    r"\btusuk\b",
    r"\bledak\b",
    r"\bhajar\b",
    r"\bserang\b",
]
AMBIGUOUS_VIOLENCE_PATTERNS = [
    r"\bpukul\s+berapa\b",
    r"\bpukul\s+\d{1,2}(\.\d{2})?\b",
    r"\bpukul\s+jam\b",
]
PII_PATTERNS = [
    r"\bnik\b",
    r"\bktp\b",
    r"\balamat\s+lengkap\b",
    r"\bnomor\s+kartu\b",
]

NEGATIVE_WORDS = {
    "sedih", "capek", "lelah", "takut", "cemas", "khawatir", "panik",
    "hancur", "gagal", "sendiri", "kesepian", "marah", "stress", "stres",
    "tertekan", "putus asa", "menangis", "buruk", "sakit",
}
POSITIVE_WORDS = {
    "senang", "lega", "tenang", "bahagia", "semangat", "baik", "aman",
    "bersyukur", "nyaman", "membaik", "kuat", "terbantu",
}
QUESTION_CUES = {
    "apa", "gimana", "bagaimana", "kenapa", "mengapa", "harus", "boleh",
    "bisa", "cara", "saran", "solusi", "menurutmu", "menurut lu",
}
ADVICE_CUES = {
    "saran", "solusi", "cara", "gimana caranya", "harus apa", "aku harus",
    "tolong bantu", "bantu aku", "kasih tips", "kasih advice",
}
GREETING_PATTERNS = [
    r"^(hai|halo|hello|hi|pagi|siang|sore|malam)(\s+(sereluna|luna))?$",
    r"^(hai|halo|hello|hi)\s+(semua|guys)$",
]
ACHIEVEMENT_WORDS = {
    "berhasil", "akhirnya", "senang", "lega", "bangga", "lulus", "selesai",
    "membaik", "diterima", "menang",
}
EMOJI_ROTATION = [
    "\U0001f331",
    "\U0001f319",
    "\U0001f642",
    "\U0001f90d",
    "\u2728",
]
BANNED_CHAT_OPENERS = [
    "Aku paham, {name}.",
    "{name}, aku merasa khawatir...",
    "Tentu, {name}!",
    "Baiklah, {name}.",
    "Aku mengerti, {name}.",
    "Wah, {name}, saya sangat prihatin.",
    "Halo lagi, {name}.",
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


def extract_keywords(text: str) -> List[str]:
    if not text:
        return []
    try:
        kw_extractor = yake.KeywordExtractor(lan="id", n=1, top=3, features=None)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except Exception:
        return []


def _normalize_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _match_patterns(text: str, patterns: List[str]) -> List[str]:
    matches: List[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(pattern)
    return matches


def _has_ambiguous_violence_context(text: str) -> bool:
    return bool(_match_patterns(text, AMBIGUOUS_VIOLENCE_PATTERNS))


def _contains_any(text: str, cues: set[str]) -> bool:
    return any(cue in text for cue in cues)


def _is_greeting_only(normalized_text: str) -> bool:
    return bool(_match_patterns(normalized_text, GREETING_PATTERNS))


def _assistant_turn_count(history_text: str) -> int:
    return len(re.findall(r"(?m)^Sereluna:", history_text or ""))


def _relationship_stage(assistant_turns: int) -> str:
    if assistant_turns <= 0:
        return "new_room"
    if assistant_turns <= 2:
        return "warming_up"
    if assistant_turns <= 7:
        return "familiar"
    return "deep_room"


def _detect_user_register(text: str, history_text: str) -> str:
    combined = _normalize_text(f"{history_text or ''} {text or ''}")
    if re.search(r"\b(gua|gue|gw|lu|lo|elo)\b", combined):
        return "gue-lu santai"
    if re.search(r"\b(aku|kamu|dirimu)\b", combined):
        return "aku-kamu hangat"
    if re.search(r"\b(saya|anda)\b", combined):
        return "saya-anda lembut"
    return "aku-kamu santai"


def _tone_guidance(stage: str, user_register: str) -> str:
    if stage == "new_room":
        return f"mulai hangat tapi tetap natural; pakai register {user_register}; jangan terlalu formal"
    if stage == "warming_up":
        return f"lanjutkan obrolan seperti sudah mulai kenal; pakai register {user_register}; kurangi sapaan pembuka"
    if stage == "familiar":
        return f"lebih santai dan responsif seperti teman yang sudah mengikuti cerita; pakai register {user_register}"
    return f"deep session: jangan reset konteks; pakai register {user_register}; respons harus terasa mengikuti alur chat panjang"


def _continuity_guidance(stage: str) -> str:
    if stage == "new_room":
        return "boleh membuka ruang ngobrol, tapi tetap langsung nyambung ke pesan user."
    if stage == "warming_up":
        return "lanjutkan dari pesan sebelumnya dan jangan memperkenalkan Sereluna ulang."
    if stage == "familiar":
        return "rujuk detail obrolan sebelumnya kalau relevan, seolah Sereluna benar-benar mengikuti sesi ini."
    return "anggap ini room yang sudah panjang; jangan pakai pembuka generik, jangan recap berlebihan, dan respons sebagai lanjutan langsung dari konteks terakhir."


def classify_chat_intent(text: str, sentiment_score: int, risk_level: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return "empty"
    if risk_level == "high":
        return "safety_support"
    if _is_greeting_only(normalized):
        return "check_in"
    if "?" in (text or "") or _contains_any(normalized, ADVICE_CUES):
        return "advice_or_problem_solving"
    if sentiment_score <= 2 or _contains_any(normalized, NEGATIVE_WORDS):
        return "emotional_support"
    if _contains_any(normalized, ACHIEVEMENT_WORDS) or sentiment_score >= 4:
        return "celebration_or_progress"
    if _contains_any(normalized, QUESTION_CUES):
        return "curious_question"
    return "reflective_companion"


def estimate_emotional_intensity(text: str, mood_signal: str, sentiment_score: int, risk_level: str) -> str:
    normalized = _normalize_text(text)
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


def build_response_style_plan(
    text: str,
    mood_signal: str,
    risk_level: str,
    sentiment_score: int,
    session_summary: str,
    history_text: str,
) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    word_count = len(normalized.split()) if normalized else 0
    assistant_turns = _assistant_turn_count(history_text)
    intent = classify_chat_intent(text, sentiment_score, risk_level)
    intensity = estimate_emotional_intensity(text, mood_signal, sentiment_score, risk_level)
    stage = _relationship_stage(assistant_turns)
    user_register = _detect_user_register(text, history_text)

    if intent == "check_in":
        desired_paragraphs = 2 if stage == "new_room" else 3
    elif intent in {"advice_or_problem_solving", "emotional_support"} or intensity in {"heavy", "tender"}:
        desired_paragraphs = 4 if stage in {"familiar", "deep_room"} or word_count >= 45 else 3
    elif intent == "safety_support":
        desired_paragraphs = 2
    else:
        desired_paragraphs = 4 if stage == "deep_room" else 3

    opening_variants = {
        "check_in": [
            "sapa balik singkat lalu ajak user cerita kondisi hari ini",
            "mulai dengan energi hangat tanpa menyebut nama user",
        ],
        "advice_or_problem_solving": [
            "jawab pertanyaan inti dulu sebelum validasi emosi",
            "petakan masalah user secara ringkas lalu beri langkah konkret",
        ],
        "emotional_support": [
            "pantulkan satu detail spesifik dari cerita user",
            "validasi rasa lelah atau berat tanpa kalimat template",
        ],
        "celebration_or_progress": [
            "ikut merayakan progres user secara natural",
            "tandai hal kecil yang layak diapresiasi",
        ],
        "safety_support": [
            "validasi kondisi berat dan arahkan ke bantuan nyata",
            "bicara tenang, langsung, dan tidak panjang berlebihan",
        ],
        "curious_question": [
            "jawab seperti teman ngobrol yang informatif",
            "beri jawaban jelas lalu kaitkan dengan konteks Sereluna",
        ],
        "reflective_companion": [
            "lanjutkan topik tanpa sapaan ulang",
            "mulai dari respons yang terasa spontan dan relevan",
        ],
    }
    options = opening_variants.get(intent, opening_variants["reflective_companion"])
    opening_strategy = options[(assistant_turns + word_count) % len(options)]

    support_moves = ["pakai bahasa Indonesia kasual yang tetap aman dan suportif"]
    if intent == "advice_or_problem_solving":
        support_moves.extend([
            "beri 2-3 langkah praktis yang bisa dicoba hari ini",
            "jelaskan alasan singkat di balik saran",
            "akhiri dengan satu pertanyaan pilihan supaya user mudah balas",
        ])
    elif intent == "emotional_support":
        support_moves.extend([
            "validasi emosi berdasarkan detail pesan, bukan diagnosis",
            "tawarkan satu micro-action ringan seperti napas, minum, atau tulis satu kalimat",
            "akhiri dengan satu pertanyaan lembut tentang bagian paling berat",
        ])
    elif intent == "celebration_or_progress":
        support_moves.extend([
            "beri apresiasi yang spesifik",
            "ajak user menyimpan pola baik yang sedang muncul",
        ])
    elif intent == "check_in":
        support_moves.extend([
            "jangan membuka obrolan dengan kalimat formal",
            "beri ruang user memilih mau cerita singkat atau panjang",
        ])
    elif intent == "safety_support":
        support_moves.extend([
            "prioritaskan keselamatan dan bantuan manusia tepercaya",
            "hindari emoji, candaan, dan instruksi yang terdengar menggurui",
        ])
    else:
        support_moves.extend([
            "respons seperti teman sebaya yang nyambung",
            "gunakan konteks memori hanya jika benar-benar relevan",
        ])

    emoji_allowed = (
        risk_level != "high"
        and intensity != "crisis"
        and intent != "safety_support"
        and (assistant_turns + word_count) % 3 != 0
    )
    emoji = EMOJI_ROTATION[(assistant_turns + len(normalized)) % len(EMOJI_ROTATION)] if emoji_allowed else None
    name_allowed = (
        assistant_turns <= 1 and intent == "check_in"
    ) or (
        intent == "celebration_or_progress" and assistant_turns % 6 == 0
    )

    return {
        "intent": intent,
        "emotional_intensity": intensity,
        "relationship_stage": stage,
        "assistant_turns": assistant_turns,
        "user_register": user_register,
        "desired_paragraphs": desired_paragraphs,
        "target_words": {
            "minimum": 120 if desired_paragraphs == 3 else 160 if desired_paragraphs == 4 else 80,
            "maximum": 260 if desired_paragraphs <= 3 else 360,
        },
        "opening_strategy": opening_strategy,
        "support_moves": support_moves,
        "tone_guidance": _tone_guidance(stage, user_register),
        "continuity_guidance": _continuity_guidance(stage),
        "name_policy": {
            "allowed": name_allowed,
            "max_mentions": 1 if name_allowed else 0,
            "instruction": "Nama user hanya boleh muncul sesekali sebagai sentuhan personal, bukan pembuka template. Jangan tulis pola seperti 'Wah, NAMA, saya sangat prihatin'.",
        },
        "emoji_policy": {
            "allowed": emoji_allowed,
            "max_count": 1 if emoji_allowed else 0,
            "suggested": emoji,
            "instruction": "Emoji boleh muncul sesekali, maksimal satu, dan jangan dipakai pada respons krisis.",
        },
        "avoid_openers": BANNED_CHAT_OPENERS,
        "question_budget": 1,
        "memory_policy": "Sebut pola dari diary/screening hanya kalau relevan dengan pesan terbaru.",
        "algorithm": {
            "name": "Sereluna Response Planner",
            "version": "1.0",
            "signals": {
                "assistant_turns": assistant_turns,
                "relationship_stage": stage,
                "user_register": user_register,
                "word_count": word_count,
                "sentiment_score": sentiment_score,
                "risk_level": risk_level,
                "has_session_summary": bool(session_summary and session_summary.strip()),
            },
        },
    }


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
            "confidence": 1.0,
        }

    current_text = _normalize_text(text)
    screening_text = _normalize_text(screening_context)
    summary_text = _normalize_text(session_summary)
    matches: List[Dict[str, Any]] = []
    score = 0

    crisis_current = _match_patterns(current_text, CRISIS_PATTERNS)
    crisis_screening = _match_patterns(screening_text, CRISIS_PATTERNS)
    crisis_summary = _match_patterns(summary_text, CRISIS_PATTERNS)
    violence_current = _match_patterns(current_text, VIOLENCE_PATTERNS)
    sexual_current = _match_patterns(current_text, [r"\bseks\b", r"\bporno\b", r"\bmesum\b"])
    pii_current = _match_patterns(current_text, PII_PATTERNS)

    if crisis_current:
        score += RISK_WEIGHTS["crisis"]
        matches.extend({"category": "crisis", "keyword": pattern, "weight": RISK_WEIGHTS["crisis"], "source": "current_text"} for pattern in crisis_current)
    if crisis_screening:
        score += RISK_WEIGHTS["crisis"]
        matches.extend({"category": "crisis", "keyword": pattern, "weight": RISK_WEIGHTS["crisis"], "source": "screening_context"} for pattern in crisis_screening)
    if crisis_summary and not crisis_current:
        score += RISK_WEIGHTS["crisis"]
        matches.extend({"category": "crisis", "keyword": pattern, "weight": RISK_WEIGHTS["crisis"], "source": "session_summary"} for pattern in crisis_summary)

    if violence_current:
        if _has_ambiguous_violence_context(current_text):
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
    elif crisis_screening:
        level = "high"
        reason = "screening_crisis_signal"
    elif score >= RISK_THRESHOLDS["high"]:
        level = "high"
        reason = "weighted_signal_score"
    elif severe_pattern.search(screening_context or ""):
        level = "high"
        reason = "severe_screening_context"
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
