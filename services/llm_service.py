import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

from services.summary_service import clean_diary_summary

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


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
    analysis_data: Dict[str, Any],
    screening_context: str,
    session_summary: str,
    profile_context: str,
    memory_context: str,
    risk_level: str,
    mood_signal: str,
    user_name: str,
    history_text: str,
    keywords: List[str],
    relevant_diary: Optional[str] = None,
) -> Dict[str, Any]:
    symptoms = analysis_data.get("detected_symptoms", [])
    category = analysis_data.get("dominant_category", "None")
    safe_user_name = (user_name or "Teman").strip() or "Teman"

    is_new_user = (
        not session_summary
        or len(session_summary.strip()) < 10
    ) and (
        not history_text
        or len(history_text.strip()) < 10
    )
    greeting_guideline = (
        "Ini adalah awal obrolan kalian. Sapa user dengan hangat dan tanyakan kabarnya hari ini."
        if is_new_user
        else "Kalian sedang melanjutkan obrolan. Jangan mengulang salam perkenalan; lanjutkan topik yang sedang berjalan."
    )

    diary_context = (
        f"Catatan diary masa lalu yang relevan: {relevant_diary}"
        if relevant_diary
        else "Tidak ada catatan diary masa lalu yang relevan."
    )
    keywords_str = ", ".join(keywords) if keywords else "N/A"
    symptoms_str = ", ".join(symptoms) if symptoms else "Tidak terdeteksi"

    fallback_reply = (
        "Aku dengerin, ya. Ceritamu terdengar penting, dan kamu tidak perlu merapikannya dulu "
        "sebelum cerita ke Sereluna. Coba mulai dari bagian yang paling berat terasa sekarang."
    )
    fallback_summary = build_fallback_session_summary(
        previous_summary=session_summary,
        user_message=user_message,
        assistant_reply=fallback_reply,
        mood_signal=mood_signal,
        risk_level=risk_level,
    )

    system_prompt = f"""Kamu adalah Sereluna, teman curhat dan asisten digital yang empatik untuk {safe_user_name}.
Gunakan bahasa Indonesia sehari-hari yang hangat, natural, dan tidak menggurui. Kamu bukan pengganti psikolog, tetapi kamu bisa memberi dukungan emosional awal dan mengarahkan user ke fitur Konselor jika perlu.
Jawaban harus cukup panjang, natural, dan terasa manusiawi. Idealnya 2-4 paragraf ringkas, kecuali user memang meminta jawaban singkat. Jangan terlalu pendek.

DATA USER & KONTEKS:
- Nama user: {safe_user_name}
- Mood signal dari aplikasi: {mood_signal or "N/A"}
- Screening context DASS-21: {screening_context or "N/A"}
- Analisis pesan terbaru: {category} ({symptoms_str})
- Risk level backend: {risk_level or "low"}
- Profile context: {_truncate(profile_context, 1000) or "N/A"}
- Kata kunci percakapan: {keywords_str}
- {diary_context}
- Memory context gabungan: {_truncate(memory_context, 5000) or "N/A"}

ATURAN:
1. {greeting_guideline}
2. Pakai konteks skrining, profil, dan ringkasan sesi hanya untuk menyesuaikan dukungan, bukan untuk memberi diagnosis.
3. Jika user bertanya soal psikolog atau konsultasi, arahkan ke menu Konselor di aplikasi.
4. Jika ada tanda bahaya, validasi perasaan user dan sarankan mencari bantuan orang tepercaya atau layanan darurat setempat.
5. Jika risk level berasal dari screening lama tetapi pesan user sekarang bersifat normal atau santai, jangan pakai respons krisis. Balas percakapan normal dengan empati.
5. Kembalikan JSON valid saja.

Schema JSON:
{{
  "reply": "jawaban Sereluna",
  "session_summary": "catatan diary singkat yang langsung berisi inti percakapan; jangan mulai dengan 'Berikut adalah ringkasan', 'Ringkasan:', atau kalimat pembuka sejenis",
  "sentiment_score": 1,
  "suggested_action": "saran singkat atau null",
  "risk_flag": false
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
            temperature=0.5,
            max_completion_tokens=900,
        )
        parsed = _parse_json_object(content, {})
        reply = (parsed.get("reply") or fallback_reply).strip()
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
        }
    except Exception:
        return {
            "reply": fallback_reply,
            "session_summary": fallback_summary,
            "sentiment_score": 3,
            "suggested_action": None,
            "risk_flag": risk_level == "high",
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
