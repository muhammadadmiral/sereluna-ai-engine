from typing import Any, Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from services.nlp.lexicons import BANNED_CHAT_OPENERS, EMOJI_ROTATION
from services.nlp.utils import normalize_text
from services.nlp.session import (
    assistant_turn_count, relationship_stage, detect_user_register, 
    tone_guidance, continuity_guidance, classify_chat_intent, 
    estimate_emotional_intensity, is_short_listener_turn
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
    normalized = normalize_text(text)
    word_count = len(normalized.split()) if normalized else 0
    assistant_turns = assistant_turn_count(history_text)
    intent = classify_chat_intent(text, sentiment_score, risk_level)
    intensity = estimate_emotional_intensity(text, mood_signal, sentiment_score, risk_level)
    stage = relationship_stage(assistant_turns)
    user_register = detect_user_register(text, history_text)
    short_listener_turn = is_short_listener_turn(text)

    desired_paragraphs = choose_desired_paragraphs(
        intent=intent,
        intensity=intensity,
        word_count=word_count,
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
        "response_feedback": [
            "akui feedback user secara santai dan langsung sesuaikan gaya",
            "jelaskan singkat kenapa tadi bisa terlalu pendek atau panjang",
        ],
        "factual_or_product_question": [
            "jawab fakta yang ditanya dulu secara jujur dan ringkas",
            "kalau tidak punya data pasti, bilang tidak punya akses angka pastinya",
        ],
        "casual_reference": [
            "tangkap referensi user secara santai",
            "jangan langsung mengubah referensi lagu/film menjadi sesi dukungan emosional berat",
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
            "kalau user baru memberi potongan cerita pendek, boleh lanjutkan dengan attentive continuer seperti 'ohh terus?' atau 'lanjut, aku ngikutin'",
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
    elif intent == "response_feedback":
        support_moves.extend([
            "jangan defensif",
            "beri komitmen gaya respons berikutnya dengan bahasa natural",
            "jangan berubah menjadi promosi aplikasi",
        ])
    elif intent == "factual_or_product_question":
        support_moves.extend([
            "jawab langsung tanpa klaim jumlah pengguna kalau datanya tidak tersedia",
            "hindari paragraf motivasi kesehatan mental yang tidak ditanya",
            "boleh sebut privasi secara singkat kalau relevan",
        ])
    elif intent == "casual_reference":
        support_moves.extend([
            "balas seperti teman yang nangkep referensi",
            "boleh tanya ringan apakah user cuma quote lagu atau lagi relate beneran",
            "jangan membuat asumsi sedih/kehilangan kalau user belum bilang",
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
        (assistant_turns <= 1 and intent == "check_in")
        or (intent == "celebration_or_progress" and assistant_turns % 6 == 0)
    )

    return {
        "intent": intent,
        "emotional_intensity": intensity,
        "relationship_stage": stage,
        "assistant_turns": assistant_turns,
        "user_register": user_register,
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
        "memory_policy": "Pakai memori lama hanya saat relevan dengan pesan terbaru. Untuk sapaan netral di room baru, abaikan diary/screening lama dan jawab seperti fresh greeting.",
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
