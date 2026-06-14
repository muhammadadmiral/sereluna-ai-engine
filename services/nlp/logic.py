from typing import Any, Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from services.nlp.lexicons import BANNED_CHAT_OPENERS, EMOJI_ROTATION
from services.nlp.utils import normalize_text
from services.nlp.session import (
    assistant_turn_count, relationship_stage, detect_user_register, 
    tone_guidance, continuity_guidance, classify_chat_intent, 
    estimate_emotional_intensity, is_short_listener_turn, build_user_style_profile
)
from services.nlp.response_policy import choose_desired_paragraphs, target_words_for_paragraphs

DIARY_RETRIEVAL_THRESHOLD = 0.1

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
            "validasi emosi tanpa kesan nge-judge atau diagnosis",
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
    normalized = normalize_text(text)
    word_count_val = len(normalized.split()) if normalized else 0
    assistant_turns = assistant_turn_count(history_text)
    intent = classify_chat_intent(text, sentiment_score, risk_level)
    intensity = estimate_emotional_intensity(text, mood_signal, sentiment_score, risk_level)
    stage = relationship_stage(assistant_turns)
    user_register = detect_user_register(text, history_text)
    short_listener_turn = is_short_listener_turn(text)

    desired_paragraphs = choose_desired_paragraphs(
        intent=intent,
        intensity=intensity,
        word_count=word_count_val,
        short_listener_turn=short_listener_turn,
    )

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
            "sapa balik santai dan tanya kabarnya hari ini gimana",
            "mulai dengan sapaan hangat yang nggak kaku",
        ],
        "advice_or_problem_solving": [
            "coba pahami dulu masalahnya sebelum kasih saran",
            "tanggapi ceritanya dengan empati, baru tawarkan solusi kecil",
        ],
        "emotional_support": [
            "tunjukkan kalau kamu dengerin detail ceritanya",
            "validasi perasaannya pake bahasa yang biasa kita pake sehari-hari",
        ],
        "celebration_or_progress": [
            "ikut seneng denger progres user",
            "kasih apresiasi buat hal kecil yang udah user lakuin",
        ],
        "safety_support": [
            "validasi kondisi berat user dengan tenang dan langsung",
            "tunjukkan kamu ada di sini untuk nemenin di masa sulit",
        ],
        "curious_question": [
            "jawab pertanyaannya kayak lagi ngobrol sama temen",
            "jelasin dikit hubungannya sama kesehatan mental kalo nyambung",
        ],
        "clarification_followup": [
            "fokus jawab apa yang user tanya barusan",
            "kalo ada salah paham, lurusin pake bahasa yang enak",
        ],
        "meta_challenge": [
            "akui koreksi user dengan santai dan terbuka",
            "cabut asumsi yang salah dan minta user jelasin maksudnya",
        ],
        "response_feedback": [
            "terima masukannya dengan senang hati",
            "janji buat sesuaikan gaya ngobrol biar makin nyambung",
        ],
        "factual_or_product_question": [
            "kasih jawaban yang jujur dan nggak muter-muter",
            "fokus ke fakta yang ditanya aja",
        ],
        "casual_reference": [
            "tanggapi referensinya dengan asik",
            "tanya dikit kenapa user kepikiran hal itu",
        ],
        "casual_banter": [
            "ikutin bercandanya user biar suasana cair",
            "balas pendek dan santai aja",
        ],
        "reflective_companion": [
            "lanjutin obrolan biar ngalir terus",
            "kasih respon yang spontan dan relevan",
        ],
    }
    options = opening_variants.get(intent, opening_variants["reflective_companion"])
    opening_strategy = options[(assistant_turns + word_count_val) % len(options)]

    support_moves = ["pake bahasa Indonesia yang asik, santai, dan penuh empati"]
    if intent == "advice_or_problem_solving":
        support_moves.extend([
            "kasih saran praktis yang gampang dicoba",
            "jelasin kenapa saran itu mungkin membantu",
            "tanya pendapat user soal saran itu",
        ])
    elif intent == "emotional_support":
        support_moves.extend([
            "validasi emosi tanpa kesan nge-judge atau diagnosis",
            "ajak tarik napas atau minum air kalo ceritanya berat",
            "pake attentive continuer kayak 'oh gitu ya...', 'terus gimana?'",
            "tanya pelan-pelan bagian mana yang paling bikin sesak",
        ])
    elif intent == "celebration_or_progress":
        support_moves.extend([
            "kasih selamat yang tulus",
            "tanya apa yang bikin user ngerasa berhasil hari ini",
        ])
    elif intent == "check_in":
        support_moves.extend([
            "hindari sapaan template yang kaku",
            "biarin user cerita apa aja yang ada di kepalanya",
        ])
    elif intent == "safety_support":
        support_moves.extend([
            "prioritaskan keselamatan dan bantuan nyata",
            "jangan pake bahasa yang kesannya nyuruh-nyuruh",
        ])
    elif intent == "response_feedback":
        support_moves.extend([
            "akui kalo respon sebelumnya kurang pas",
            "langsung balik ke topik utama user",
        ])
    elif intent == "clarification_followup":
        support_moves.extend([
            "jawab pertanyaan spesifik user dengan singkat",
            "nggak usah bahas masalah emosi kalo user lagi tanya teknis",
        ])
    elif intent == "meta_challenge":
        support_moves.extend([
            "akui kalo kamu tadi 'sok tahu'",
            "ajak user buat koreksi pemahamanmu",
        ])
    elif intent == "factual_or_product_question":
        support_moves.extend([
            "jawab langsung inti pertanyaannya",
            "sebutin soal privasi cuma kalo relevan",
        ])
    elif intent == "casual_reference":
        support_moves.extend([
            "balas kayak temen yang nangkep referensinya",
            "jangan langsung ditarik ke topik sedih kalo user cuma lagi denger lagu",
        ])
    elif intent == "casual_banter":
        support_moves.extend([
            "bercanda balik dikit biar seru",
            "jangan kaku nanggapi candaan user",
        ])
    else:
        support_moves.extend([
            "ngobrol kayak temen sebaya yang nyambung",
            "pake memori lama cuma kalo bener-bener pas",
        ])

    emoji_allowed = (
        risk_level != "high"
        and intensity != "crisis"
        and intent != "safety_support"
        and (assistant_turns + word_count_val) % 3 != 0
    )
    emoji = EMOJI_ROTATION[(assistant_turns + len(normalized)) % len(EMOJI_ROTATION)] if emoji_allowed else None
    name_allowed = (
        (assistant_turns <= 1 and intent == "check_in")
        or (intent == "celebration_or_progress" and assistant_turns % 6 == 0)
    )

    user_style_profile = build_user_style_profile(text, history_text)
    response_mode = "direct_response"
    if intent == "check_in":
        response_mode = "contextual_check_in" if assistant_turns > 0 or session_summary.strip() else "low_signal_greeting"
    elif intent == "safety_support" or risk_level == "high":
        response_mode = "crisis_response"
    elif intent == "off_domain_redirect":
        response_mode = "boundary_redirect"
    elif intent in {"emotional_support", "advice_or_problem_solving"}:
        response_mode = "assessment_response"
    elif intent in {
        "casual_banter",
        "casual_reference",
        "factual_or_product_question",
        "meta_challenge",
        "clarification_followup",
        "response_feedback",
        "celebration_or_progress",
        "curious_question",
        "reflective_companion",
    }:
        response_mode = "direct_response"

    return {
        "intent": intent,
        "response_mode": response_mode,
        "emotional_intensity": intensity,
        "relationship_stage": stage,
        "assistant_turns": assistant_turns,
        "user_register": user_register,
        "user_style_profile": user_style_profile,
        "desired_paragraphs": desired_paragraphs,
        "target_words": target_words_for_paragraphs(desired_paragraphs),
        "opening_strategy": opening_strategy,
        "support_moves": support_moves,
        "tone_guidance": tone_guidance(stage, user_register),
        "continuity_guidance": continuity_guidance(stage),
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
        "short_listener_turn": short_listener_turn,
        "memory_scope": memory_scope,
        "memory_policy": "Pakai memori lama hanya saat relevan dengan pesan terbaru. Untuk sapaan netral di room baru, abaikan diary/screening lama and jawab seperti fresh greeting.",
        "context_priority": (
            "Untuk meta challenge and follow-up pendek: pesan user terbaru + 3-6 pesan terakhir adalah sumber utama. "
            "Session summary hanya latar. Diary/screening lama hanya boleh dipakai kalau user jelas merujuk ke topik itu."
        ),
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
