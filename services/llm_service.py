import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

from services.summary_service import clean_diary_summary

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _completion(
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, str]] = None,
    temperature: float = 0.4,
    max_completion_tokens: Optional[int] = None,
) -> str:
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")

    client = Groq(api_key=api_key)
    kwargs: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if response_format:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _parse_json_object(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return fallback

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else fallback
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                return parsed if isinstance(parsed, dict) else fallback
            except json.JSONDecodeError:
                pass
    return fallback


def _truncate(value: Optional[str], limit: int) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _coerce_score(value: Any, default: int = 3) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(score, 5))


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "ya"}
    return default


def _coerce_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    return text


_EMOJI_PATTERN = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def _format_style_plan(style_plan: Optional[Dict[str, Any]], user_name: str) -> str:
    if not style_plan:
        return "Gunakan gaya natural dan mengalir santai."

    support_moves = style_plan.get("support_moves") or []
    moves_text = ", ".join(support_moves[:2]) if support_moves else "respons natural"
    
    return f"""Target respons: {style_plan.get("desired_paragraphs", 2)} paragraf santai.
Tone: {style_plan.get("tone_guidance", "hangat dan kasual")}.
Fokus strategi: {style_plan.get("opening_strategy", "langsung respons inti pesan")} ({moves_text})."""


def _format_care_intelligence(
    emotion_profile: Optional[Dict[str, Any]],
    cognitive_distortions: Optional[Dict[str, Any]],
    coping_pathway: Optional[Dict[str, Any]],
) -> str:
    emotion_profile = emotion_profile or {}
    coping_pathway = coping_pathway or {}

    primary_emotion = emotion_profile.get("primary_emotion", "neutral")
    pathway_steps = coping_pathway.get("steps") or ["Respons natural sesuai konteks."]
    step_text = ", ".join(pathway_steps[:2])
    
    return f"""Analisis Emosi Backend: {primary_emotion}.
Saran Pendekatan: {step_text}."""


def _strip_repetitive_openers(reply: str, user_name: str) -> str:
    text = (reply or "").strip()
    if not text:
        return text

    name = re.escape((user_name or "").strip())
    if name and len(name) > 1:
        text = re.sub(rf"^\s*(?:wah|aduh|duh)\s*,?\s*{name}\s*[,!.]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(rf"^\s*{name}\s*[,!.]\s*", "", text, flags=re.IGNORECASE)

    opener_pattern = (
        r"^\s*(?:aku\s+(?:paham|mengerti|ngerti|dengerin)|"
        r"tentu|baiklah|baik|oke|okay|siap)\s*[,!.]\s*"
    )
    text = re.sub(opener_pattern, "", text, count=1, flags=re.IGNORECASE).lstrip()
    text = re.sub(
        r"^\s*(?:wah|aduh|duh)\s*,?\s*(?:saya\s+sangat\s+prihatin|aku\s+ikut\s+khawatir)\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).lstrip()
    text = re.sub(
        r"^\s*(?:saya\s+sangat\s+prihatin\s+(?:mendengar|dengar)?|"
        r"aku\s+(?:merasa\s+)?khawatir\s+(?:mendengar|dengar)?)\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).lstrip()

    if name and len(name) > 1:
        text = re.sub(rf"^\s*{name}\s*[,!.]\s*", "", text, count=1, flags=re.IGNORECASE)

    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text or reply.strip()


def _limit_name_mentions(reply: str, user_name: str, max_mentions: int) -> str:
    name = (user_name or "").strip()
    if not name or len(name) <= 1:
        return reply

    pattern = re.compile(rf"\b{re.escape(name)}\b", flags=re.IGNORECASE)
    matches = list(pattern.finditer(reply))
    if len(matches) <= max_mentions:
        return reply

    text = reply
    for match in reversed(matches[max_mentions:]):
        text = text[: match.start()] + text[match.end() :]
    if max_mentions == 0 and matches:
        text = pattern.sub("", text)

    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?]){2,}", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^\s*[,!.]\s*", "", text)
    return text.strip() or reply


def _shape_paragraphs(reply: str, desired_paragraphs: int) -> str:
    text = (reply or "").strip()
    if desired_paragraphs <= 1 or "\n\n" in text or "\n-" in text or "\n1." in text:
        return text

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if len(sentences) < desired_paragraphs:
        return text

    group_size = max(1, (len(sentences) + desired_paragraphs - 1) // desired_paragraphs)
    paragraphs = [
        " ".join(sentences[index : index + group_size]).strip()
        for index in range(0, len(sentences), group_size)
    ]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _maybe_add_planned_emoji(reply: str, style_plan: Optional[Dict[str, Any]]) -> str:
    if not style_plan:
        return reply

    emoji_policy = style_plan.get("emoji_policy") or {}
    if not emoji_policy.get("allowed") or not emoji_policy.get("max_count"):
        return reply
    if _EMOJI_PATTERN.search(reply):
        return reply

    emoji = emoji_policy.get("suggested")
    if not emoji:
        return reply

    paragraphs = reply.split("\n\n")
    if not paragraphs:
        return reply
    paragraphs[0] = paragraphs[0].rstrip() + f" {emoji}"
    return "\n\n".join(paragraphs)


def _polish_sereluna_reply(reply: str, user_name: str, style_plan: Optional[Dict[str, Any]]) -> str:
    text = _strip_repetitive_openers(reply, user_name)
    name_policy = (style_plan or {}).get("name_policy") or {}
    text = _limit_name_mentions(text, user_name, int(name_policy.get("max_mentions", 0) or 0))
    text = _shape_paragraphs(text, int((style_plan or {}).get("desired_paragraphs", 2) or 2))
    text = _maybe_add_planned_emoji(text, style_plan)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() or reply.strip()


def build_fallback_session_summary(
    previous_summary: Optional[str],
    user_message: Optional[str],
    assistant_reply: Optional[str],
    mood_signal: Optional[str] = "",
    risk_level: Optional[str] = "low",
) -> str:
    parts: List[str] = []
    if previous_summary and previous_summary.strip():
        parts.append(clean_diary_summary(_truncate(previous_summary, 900)))
    if user_message and user_message.strip():
        parts.append(f"User menyampaikan: {_truncate(user_message, 260)}")
    if assistant_reply and assistant_reply.strip():
        parts.append(f"Sereluna merespons dengan dukungan: {_truncate(assistant_reply, 260)}")
    if mood_signal or risk_level:
        parts.append(f"Kondisi terakhir terbaca mood {mood_signal or 'N/A'} dengan risiko {risk_level or 'low'}.")

    return clean_diary_summary(" ".join(part for part in parts if part).strip()) or "Belum ada cukup percakapan untuk dirangkum."


def analyze_symptoms_llm(user_message: str) -> Dict[str, Any]:
    dass_reference = """
    Reference DASS-21 Indicators:
    1. DEPRESSION: Hopelessness, devaluation of life, self-deprecation, lack of interest, anhedonia.
    2. ANXIETY: Autonomic arousal, skeletal muscle effects, situational anxiety, anxious affect.
    3. STRESS: Difficulty relaxing, nervous arousal, easily upset/agitated, irritable/over-reactive.
    """
    system_prompt = (
        "ROLE: Psychological Screening Assistant.\n"
        "TASK: Analyze 'User Input' and map strictly to 'Reference DASS-21'.\n"
        f"{dass_reference}\n\n"
        "INSTRUCTION:\n"
        "- Identify symptoms present in the text.\n"
        "- Return JSON ONLY.\n"
        "- Schema: {\"detected_symptoms\": [\"string\"], "
        "\"dominant_category\": \"Depression\" | \"Anxiety\" | \"Stress\" | \"None\" | \"Mixed\"}"
    )
    user_prompt = f"USER INPUT: '{user_message or ''}'"
    fallback = {"detected_symptoms": [], "dominant_category": "None"}

    try:
        content = _completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        parsed = _parse_json_object(content, fallback)
        return {
            "detected_symptoms": parsed.get("detected_symptoms") or [],
            "dominant_category": parsed.get("dominant_category") or "None",
        }
    except Exception:
        return fallback


def generate_dialog(
    user_message: str,
    screening_context: str,
    session_summary: str,
    profile_context: str,
    memory_context: str,
    recent_daily_context: str,
    risk_level: str,
    mood_signal: str,
    user_name: str,
    history_text: str,
    keywords: List[str],
    relevant_diary: Optional[str] = None,
    style_plan: Optional[Dict[str, Any]] = None,
    emotion_profile: Optional[Dict[str, Any]] = None,
    cognitive_distortions: Optional[Dict[str, Any]] = None,
    coping_pathway: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    safe_user_name = (user_name or "Teman").strip() or "Teman"
    assistant_turns = int((style_plan or {}).get("assistant_turns", 0) or 0)

    is_new_user = (
        assistant_turns <= 0
        and (
        not session_summary
        or len(session_summary.strip()) < 10
        )
    )
    greeting_guideline = (
        "Ini adalah awal obrolan kalian. Kalau user hanya menyapa, balas sapaan dengan natural dan jangan menebak emosi dari riwayat lama."
        if is_new_user
        else "Kalian sedang melanjutkan obrolan di room yang sama. Jangan mengulang salam perkenalan; langsung lanjutkan topik dan emosi yang sedang berjalan."
    )

    diary_context = (
        f"Catatan diary masa lalu yang relevan: {relevant_diary}"
        if relevant_diary
        else "Tidak ada catatan diary masa lalu yang relevan."
    )
    keywords_str = ", ".join(keywords) if keywords else "N/A"
    style_plan_text = _format_style_plan(style_plan, safe_user_name)
    care_intelligence_text = _format_care_intelligence(
        emotion_profile=emotion_profile,
        cognitive_distortions=cognitive_distortions,
        coping_pathway=coping_pathway,
    )

    fallback_reply = (
        "Aku dengerin, ya. Ceritamu nggak harus rapi dulu buat bisa mulai dibahas di sini. "
        "Kalau sekarang rasanya penuh, kita bisa ambil satu bagian yang paling kerasa berat dan pelan-pelan urai bareng."
    )
    fallback_summary = build_fallback_session_summary(
        previous_summary=session_summary,
        user_message=user_message,
        assistant_reply=fallback_reply,
        mood_signal=mood_signal,
        risk_level=risk_level,
    )

    system_prompt = f"""Kamu adalah Sereluna, teman bicara kesehatan mental untuk {safe_user_name}.
Sereluna bukan sekadar chatbot umum: kamu memakai sinyal mood, ringkasan diary, screening DASS-21, risk classifier, dan planner respons untuk memberi dukungan yang terasa personal, kontekstual, dan aman. Kamu bukan pengganti psikolog, tetapi kamu bisa memberi dukungan emosional awal, bantu user mengurai pikiran, dan mengarahkan user ke fitur Konselor jika perlu.

GAYA SERELUNA:
- Bahasa Indonesia sehari-hari, hangat, luwes, dan boleh sedikit seperti teman sebaya, tapi tetap sensitif.
- Jawaban terasa seperti obrolan AI companion yang pintar: nyambung, spesifik ke cerita user, tidak kaku, dan tidak menggurui.
- Prioritaskan jawaban panjang yang enak dibaca: ikuti target paragraf dan target kata dari planner. Jangan menjawab satu paragraf pendek kecuali user cuma menyapa sangat singkat.
- Kalau respons terasa belum memenuhi target panjang, kembangkan dengan insight, contoh konkret, atau langkah kecil yang relevan; jangan mengulang kalimat validasi yang sama.
- Kalau user sedang cerita panjang/curhat, respons harus terasa hadir dan mengikuti alur: pantulkan detail, uraikan makna, beri opsi langkah kecil, lalu ajak lanjut.
- Kalau user baru memberi potongan cerita singkat, boleh lebih sederhana seperti teman yang mendengarkan, misalnya mengundang lanjut cerita tanpa memaksa.
- Makin panjang room chat, makin santai dan makin kontekstual. Jangan bersikap seperti baru kenal kalau riwayat chat sudah ada.
- Ikuti register user. Kalau user biasa pakai "gua/lu", boleh balas lebih santai; kalau user pakai "aku/kamu", gunakan aku-kamu hangat.
- Context timing itu penting. Jangan membawa konflik, mood buruk, atau diary lama ke sapaan netral seperti "halo guys" kecuali user sendiri mengaitkannya.
- Jangan membuka tiap balasan dengan "Aku paham", "Tentu", "Baiklah", atau menyebut nama user. Nama user maksimal sesuai planner.
- Jangan pakai gaya "Wah, NAMA, saya sangat prihatin" karena terdengar kaku dan template.
- Hindari template konseling yang berulang. Validasi harus spesifik ke detail pesan user.
- Emoji boleh sesekali sesuai planner, maksimal satu, dan jangan dipakai untuk situasi krisis.
- Jangan kebanyakan pertanyaan. Maksimal satu pertanyaan yang paling membantu untuk lanjut ngobrol.

DATA USER & KONTEKS (BACA TAPI JANGAN TERLALU KAKU):
- Nama user: {safe_user_name}
- Mood signal: {mood_signal or "N/A"}
- Risk level: {risk_level or "low"}
- Konteks 3 hari terakhir: {_truncate(recent_daily_context, 1200) or "N/A"}
- Kata kunci percakapan: {keywords_str}
- {diary_context}
- Memory context gabungan: {_truncate(memory_context, 2500) or "N/A"}

RESPONSE PLANNER (JADIKAN PANDUAN FLEKSIBEL):
{style_plan_text}

SERELUNA CARE INTELLIGENCE:
{care_intelligence_text}

ATURAN:
1. {greeting_guideline}
2. Pakai konteks skrining, profil, dan ringkasan sesi hanya untuk menyesuaikan dukungan, bukan untuk memberi diagnosis.
3. Jika user bertanya soal psikolog atau konsultasi, arahkan ke menu Konselor di aplikasi.
4. Jika ada tanda bahaya, validasi perasaan user dan sarankan mencari bantuan orang tepercaya atau layanan darurat setempat.
5. Jika risk level berasal dari screening lama tetapi pesan user sekarang bersifat normal atau santai, jangan pakai respons krisis. Balas percakapan normal dengan empati.
6. Kembalikan JSON valid saja.

Schema JSON:
{{
  "reply": "jawaban Sereluna",
  "session_summary": "catatan diary singkat yang langsung berisi inti percakapan; jangan mulai dengan 'Berikut adalah ringkasan', 'Ringkasan:', atau kalimat pembuka sejenis",
  "sentiment_score": 1,
  "suggested_action": "saran aksi nyata singkat atau null",
  "risk_flag": false,
  "detected_symptoms": ["gejala 1", "gejala 2"],
  "dominant_category": "None"
}}"""

    user_prompt = f"""Ringkasan sesi sebelumnya:
{_truncate(session_summary, 1500) or "N/A"}

Riwayat chat mentah sesi ini:
{_truncate(history_text, 6000) or "N/A"}

Pesan user sekarang:
{user_message or ""}"""

    try:
        content = _completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.72,
            max_completion_tokens=1800,
        )
        parsed = _parse_json_object(content, {})
        reply = _polish_sereluna_reply((parsed.get("reply") or fallback_reply).strip(), safe_user_name, style_plan)
        raw_next_summary = (
            parsed.get("session_summary")
            or build_fallback_session_summary(
                previous_summary=session_summary,
                user_message=user_message,
                assistant_reply=reply,
                mood_signal=mood_signal,
                risk_level=risk_level,
            )
        )
        next_summary = clean_diary_summary(raw_next_summary)

        return {
            "reply": reply,
            "session_summary": next_summary,
            "sentiment_score": _coerce_score(parsed.get("sentiment_score"), 3),
            "suggested_action": _coerce_optional_text(parsed.get("suggested_action")),
            "risk_flag": _coerce_bool(parsed.get("risk_flag"), risk_level == "high"),
            "detected_symptoms": parsed.get("detected_symptoms", []),
            "dominant_category": parsed.get("dominant_category", "None"),
        }
    except Exception:
        return {
            "reply": fallback_reply,
            "session_summary": fallback_summary,
            "sentiment_score": 3,
            "suggested_action": None,
            "risk_flag": risk_level == "high",
            "detected_symptoms": [],
            "dominant_category": "None",
        }


def _fallback_final_summary(session_raw: str, session_summary: str, user_name: str) -> str:
    if session_summary and session_summary.strip():
        return clean_diary_summary(session_summary)
    if session_raw and session_raw.strip():
        return clean_diary_summary(f"Percakapan membahas {_truncate(session_raw, 650)}")
    return "Sesi selesai, tetapi belum ada cukup percakapan untuk dirangkum."


def generate_summary(
    session_raw: str,
    session_summary: str,
    user_name: str,
) -> str:
    safe_user_name = (user_name or "Teman").strip() or "Teman"
    system_prompt = (
        "Tugasmu adalah membuat final diary summary dari sesi chat Sereluna. "
        "Tulis 3-4 kalimat dalam bahasa Indonesia. Langsung mulai dari inti cerita user. "
        "Jangan memakai pembuka seperti 'Berikut adalah ringkasan', 'Ringkasan:', "
        "atau 'Berikut adalah ringkasan dari sesi chat Sereluna dengan ...'. "
        "Jangan menyebut nama user hanya untuk membuka summary. Rangkum emosi utama user, "
        "masalah yang dibahas, dukungan yang diberikan, dan tindak lanjut yang relevan. "
        "Jangan memberi diagnosis klinis."
    )
    user_prompt = (
        f"Nama: {safe_user_name}\n"
        f"Rolling summary terakhir: {session_summary or 'N/A'}\n\n"
        f"Teks percakapan lengkap sesi ini:\n{session_raw or 'N/A'}"
    )

    try:
        content = _completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_completion_tokens=500,
        )
        cleaned = clean_diary_summary(content)
        return cleaned or _fallback_final_summary(session_raw, session_summary, safe_user_name)
    except Exception:
        return _fallback_final_summary(session_raw, session_summary, safe_user_name)
