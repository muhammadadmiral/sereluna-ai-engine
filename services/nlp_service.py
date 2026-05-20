import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import yake
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from services.nlp_lexicons import (
    ACHIEVEMENT_WORDS,
    ADVICE_CUES,
    AMBIGUOUS_VIOLENCE_PATTERNS,
    BANNED_CHAT_OPENERS,
    COGNITIVE_DISTORTION_PATTERNS,
    CRISIS_PATTERNS,
    DASS21_INDEXES,
    DASS21_THRESHOLDS,
    DISTORTION_LABELS,
    DISTORTION_REFRAME_TARGETS,
    EMOJI_ROTATION,
    EMOTION_LEXICON,
    EMOTION_LEXICON_ENTRIES,
    EMOTION_WEIGHTS,
    GREETING_PATTERNS,
    MOOD_TO_EMOTION,
    NEGATIVE_WORDS,
    PII_PATTERNS,
    POSITIVE_WORDS,
    QUESTION_CUES,
    RISK_THRESHOLDS,
    RISK_WEIGHTS,
    SEXUAL_PATTERNS,
    VIOLENCE_PATTERNS,
)

DIARY_RETRIEVAL_THRESHOLD = 0.1


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


def _phrase_score(normalized_text: str, phrase: str) -> int:
    if phrase not in normalized_text:
        return 0
    return 2 if " " in phrase else 1


@lru_cache(maxsize=1)
def _emotion_centroid_model() -> Dict[str, Any]:
    training_rows = [
        {
            "term": row["term"],
            "emotion": row["emotion"],
            "weight": int(row.get("weight") or 1),
        }
        for row in EMOTION_LEXICON_ENTRIES
        if row.get("term") and row.get("emotion")
    ]
    terms = [row["term"] for row in training_rows]
    labels = [row["emotion"] for row in training_rows]
    weights = np.array([max(1, row["weight"]) for row in training_rows], dtype=float)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True)
    vectors = normalize(vectorizer.fit_transform(terms))

    centroids: Dict[str, Any] = {}
    for emotion in sorted(set(labels)):
        indexes = [index for index, label in enumerate(labels) if label == emotion]
        if not indexes:
            continue
        class_weights = weights[indexes]
        class_vectors = vectors[indexes].multiply(class_weights[:, None])
        centroid = sparse.csr_matrix(class_vectors.sum(axis=0)) / class_weights.sum()
        centroids[emotion] = normalize(centroid)

    return {
        "vectorizer": vectorizer,
        "centroids": centroids,
        "training_rows": len(training_rows),
        "classes": sorted(centroids),
    }


def classify_emotion_ml(text: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    if not normalized:
        return {
            "predicted_emotion": "neutral",
            "confidence": 0.0,
            "scores": {},
            "algorithm": {
                "name": "TF-IDF Nearest-Centroid Emotion Classifier",
                "version": "1.0",
                "training_source": "data/lexicons/emotion_lexicon.csv",
            },
        }

    model = _emotion_centroid_model()
    query_vector = normalize(model["vectorizer"].transform([normalized]))
    scores = {
        emotion: float(cosine_similarity(query_vector, centroid)[0][0])
        for emotion, centroid in model["centroids"].items()
    }
    predicted_emotion, confidence = max(scores.items(), key=lambda item: item[1])
    if confidence < 0.02:
        predicted_emotion = "neutral"

    return {
        "predicted_emotion": predicted_emotion,
        "confidence": round(float(confidence), 4),
        "scores": {emotion: round(score, 4) for emotion, score in sorted(scores.items())},
        "algorithm": {
            "name": "TF-IDF Nearest-Centroid Emotion Classifier",
            "version": "1.0",
            "method": "fit TF-IDF character n-gram vectors from curated emotion lexicon, then classify by cosine distance to class centroids",
            "training_source": "data/lexicons/emotion_lexicon.csv",
            "training_rows": model["training_rows"],
            "classes": model["classes"],
        },
    }


def build_emotion_profile(text: str, mood_signal: str, sentiment_score: int, risk_level: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
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
    normalized = _normalize_text(text)
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


def select_coping_pathway(
    text: str,
    risk_level: str,
    sentiment_score: int,
    emotion_profile: Dict[str, Any],
    distortion_profile: Dict[str, Any],
) -> Dict[str, Any]:
    primary_emotion = emotion_profile.get("primary_emotion", "neutral")
    intensity = emotion_profile.get("intensity", "neutral")
    distortion_count = int(distortion_profile.get("count", 0) or 0)
    intent = classify_chat_intent(text, sentiment_score, risk_level)

    if risk_level == "high":
        pathway = "safety_triage"
        steps = [
            "validasi kondisi berat secara singkat",
            "dorong user menghubungi orang tepercaya atau layanan darurat setempat",
            "arahkan ke fitur Konselor",
        ]
    elif distortion_count:
        pathway = "cbt_reframe_plus_problem_solving"
        steps = [
            "validasi emosi tanpa menguatkan pikiran katastrofik",
            "ajak user membedakan fakta, asumsi, dan skenario terburuk",
            "beri satu langkah problem-solving yang konkret",
        ]
    elif primary_emotion == "anxiety":
        pathway = "grounding_then_plan"
        steps = [
            "turunkan arousal dengan grounding singkat",
            "pecah kekhawatiran menjadi hal yang bisa dikontrol dan tidak bisa dikontrol",
            "pilih satu aksi kecil berikutnya",
        ]
    elif primary_emotion in {"sadness", "loneliness"}:
        pathway = "emotional_validation_and_connection"
        steps = [
            "validasi rasa sedih atau sendiri secara spesifik",
            "ajak user menyebut kebutuhan emosional yang belum terpenuhi",
            "sarankan dukungan ringan dari orang aman atau journaling",
        ]
    elif primary_emotion == "anger":
        pathway = "deescalation_and_boundary"
        steps = [
            "akui rasa kesal tanpa memperbesar konflik",
            "beri jeda regulasi emosi",
            "bantu susun batasan atau kalimat respons yang lebih aman",
        ]
    elif primary_emotion == "fatigue":
        pathway = "low_energy_next_step"
        steps = [
            "turunkan target menjadi langkah paling kecil",
            "sarankan istirahat mikro atau prioritas satu tugas",
            "hindari nasihat yang menambah beban",
        ]
    elif intent == "advice_or_problem_solving":
        pathway = "structured_problem_solving"
        steps = [
            "jawab inti pertanyaan user",
            "urai opsi dengan konsekuensi singkat",
            "beri langkah praktis yang bisa dicoba hari ini",
        ]
    else:
        pathway = "reflective_companionship"
        steps = [
            "lanjutkan obrolan dari konteks terakhir",
            "pantulkan satu detail penting",
            "beri ruang user memilih arah cerita berikutnya",
        ]

    return {
        "pathway": pathway,
        "intent": intent,
        "intensity": intensity,
        "steps": steps,
        "algorithm": {
            "name": "Sereluna Coping Pathway Decision Tree",
            "version": "1.0",
            "inputs": ["risk_level", "sentiment_score", "primary_emotion", "cognitive_distortion_count", "chat_intent"],
        },
    }


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

    if stage == "new_room" and intent == "check_in":
        memory_scope = "current_room_only"
    elif stage == "new_room":
        memory_scope = "current_room_plus_relevant_diary_only"
    elif stage == "warming_up":
        memory_scope = "current_room_first"
    else:
        memory_scope = "current_room_plus_relevant_memory"

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
            "minimum": 180 if desired_paragraphs == 3 else 260 if desired_paragraphs == 4 else 90,
            "maximum": 220 if desired_paragraphs == 2 else 360 if desired_paragraphs == 3 else 520,
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
        "memory_scope": memory_scope,
        "memory_policy": "Pakai memori lama hanya saat relevan dengan pesan terbaru. Untuk sapaan netral di room baru, abaikan diary/screening lama dan jawab seperti fresh greeting.",
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
                "memory_scope": memory_scope,
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
    sexual_current = _match_patterns(current_text, SEXUAL_PATTERNS)
    pii_current = _match_patterns(current_text, PII_PATTERNS)

    if crisis_current:
        score += RISK_WEIGHTS["crisis"]
        matches.extend({"category": "crisis", "keyword": pattern, "weight": RISK_WEIGHTS["crisis"], "source": "current_text"} for pattern in crisis_current)
    if crisis_screening:
        matches.extend({"category": "crisis", "keyword": pattern, "weight": 0, "source": "screening_context"} for pattern in crisis_screening)
    if crisis_summary and not crisis_current:
        matches.extend({"category": "crisis", "keyword": pattern, "weight": 1, "source": "session_summary"} for pattern in crisis_summary)

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
    current_is_greeting = _is_greeting_only(_normalize_text(text))
    risk = classify_risk(
        text=text,
        screening_context="" if current_is_greeting else screening_context,
        session_summary="" if current_is_greeting else session_summary,
    )
    retrieval = (
        {"diary": None, "similarity": 0.0, "index": None, "threshold": DIARY_RETRIEVAL_THRESHOLD}
        if current_is_greeting
        else find_relevant_diary_with_score(text, past_diaries)
    )
    keywords = extract_keywords(text)
    sentiment_score = calculate_sentiment_score(text, mood_signal)
    emotion_profile = build_emotion_profile(text, mood_signal, sentiment_score, risk["level"])
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
        else classify_emotion_ml(text)
    )
    emotion_profile["ml_prediction"] = ml_emotion
    if emotion_profile["primary_emotion"] in {"neutral", "distress"} and ml_emotion["predicted_emotion"] != "neutral":
        emotion_profile["primary_emotion"] = ml_emotion["predicted_emotion"]
        emotion_profile["intensity"] = "low"
    distortion_profile = detect_cognitive_distortions(text)
    coping_pathway = select_coping_pathway(
        text=text,
        risk_level=risk["level"],
        sentiment_score=sentiment_score,
        emotion_profile=emotion_profile,
        distortion_profile=distortion_profile,
    )

    return {
        "risk_level": risk["level"],
        "risk": risk,
        "sentiment_score": sentiment_score,
        "keywords": keywords,
        "relevant_diary": retrieval["diary"],
        "retrieval": retrieval,
        "emotion_profile": emotion_profile,
        "ml_emotion_classifier": ml_emotion,
        "cognitive_distortions": distortion_profile,
        "coping_pathway": coping_pathway,
        "algorithms": {
            "main": [
                "weighted_rule_based_risk_classification",
                "tfidf_cosine_similarity_diary_retrieval",
                "emotion_lexicon_intensity_profile",
                "tfidf_nearest_centroid_emotion_classifier",
                "cognitive_distortion_pattern_mining",
                "coping_pathway_decision_tree",
            ],
            "supporting": ["lexicon_based_sentiment_scoring", "yake_keyword_extraction"],
        },
    }
