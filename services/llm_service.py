import os
import json
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant"

def analyze_symptoms_llm(user_message: str) -> Dict[str, Any]:
    dass_reference = """
    Reference DASS-21 Indicators:
    1. DEPRESSION: Hopelessness, devaluation of life, self-deprecation, lack of interest, anhedonia.
    2. ANXIETY: Autonomic arousal, skeletal muscle effects, situational anxiety, anxious affect.
    3. STRESS: Difficulty relaxing, nervous arousal, easily upset/agitated, irritable/over-reactive.
    """
    system_prompt = (
        f"ROLE: Psychological Screening Assistant.\n"
        f"TASK: Analyze 'User Input' and map strictly to 'Reference DASS-21'.\n"
        f"{dass_reference}\n\n"
        f"INSTRUCTION:\n"
        f"- Identify symptoms present in the text.\n"
        f"- Return JSON ONLY.\n"
        f"- Schema: {{ 'detected_symptoms': ['string'], 'dominant_category': 'Depression' | 'Anxiety' | 'Stress' | 'None' | 'Mixed' }}"
    )
    user_prompt = f"USER INPUT: '{user_message}'"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"detected_symptoms": [], "dominant_category": "None"}

def generate_dialog(
    user_message: str,
    analysis_data: Dict[str, Any],
    screening_context: str,
    session_summary: str,
    profile_context: str,
    risk_level: str,
    mood_signal: str,
    user_name: str,
    history_text: str,
    keywords: List[str],
    relevant_diary: Optional[str] = None
) -> Dict[str, Any]:
    symptoms = analysis_data.get("detected_symptoms", [])
    category = analysis_data.get("dominant_category", "None")
    
    is_new_user = (not session_summary or len(session_summary.strip()) < 10) and (not history_text or len(history_text.strip()) < 10)
    greeting_guideline = (
        "Ini adalah awal obrolan kalian. Sapa user dengan hangat, tanyakan bagaimana perasaannya hari ini."
        if is_new_user else
        "Kalian sedang berada di tengah-tengah obrolan yang sudah berjalan. JANGAN mengulang salam perkenalan (seperti Halo/Selamat pagi/malam). Lanjutkan saja topik yang sedang dibahas."
    )

    diary_context = f"Catatan Diary Masa Lalu yang Relevan: {relevant_diary}" if relevant_diary else "Tidak ada catatan diary masa lalu yang relevan."
    keywords_str = ", ".join(keywords) if keywords else "N/A"

    system_prompt = f"""Kamu adalah Sereluna, teman curhat dan asisten digital yang sangat empatik dan supportif untuk {user_name}.
Gaya Bicara: Casual, hangat, layaknya sahabat dekat yang mengerti kondisinya. Gunakan bahasa Indonesia sehari-hari yang luwes. Jangan kaku. Berbicaralah dalam beberapa paragraf agar terasa lebih niat dan panjang (jangan hanya 1 paragraf pendek).

DATA USER & KONTEKS:
- Skor DASS-21: {category} ({", ".join(symptoms)})
- Status Sesi: {"Awal Chat (User Baru/Sesi Baru)" if is_new_user else "Melanjutkan Obrolan Lama"}
- Kata Kunci Percakapan: {keywords_str}
- {diary_context}

ATURAN WAJIB:
1. {greeting_guideline}
2. Bicaralah panjang lebar dan komprehensif, tunjukkan empati yang mendalam. Buat user merasa benar-benar didengarkan.
3. Jika ditanya soal psikolog/konsultasi: Beritahu bahwa Sereluna menyediakan menu "Konselor" di dalam aplikasi dengan konselor khusus yang siap membantu. Arahkan user untuk mencoba menu tersebut.
4. Jika user toxic/kasar: Tetap sabar dan asik, tanya kenapa dia marah tanpa menceramahi.
5. JAWAB DALAM FORMAT JSON: {{"reply": "jawaban panjangmu disini (bisa pakai \\n untuk paragraf baru)", "sentiment_score": 1-5, "suggested_action": "saran singkat", "risk_flag": true/false}}"""

    user_prompt = f"""Konteks Profil: {profile_context or "N/A"}
Konteks Skrining DASS: {screening_context or "N/A"}
Konteks Chat/Diary Sebelumnya: {session_summary or "N/A"}
Riwayat Chat (Terkini):
{history_text or "Belum ada riwayat"}

PESAN USER SEKARANG: "{user_message}" """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "reply": f"Duh, Sereluna lagi pusing nih mikirin kalimat yang pas. Boleh bantu ketik ulang curhatanmu? (Error: {str(e)[:40]})",
            "sentiment_score": 3,
            "suggested_action": None,
            "risk_flag": False
        }

def generate_summary(session_raw: str, session_summary: str, user_name: str) -> str:
    system_prompt = "Tugasmu adalah merangkum percakapan untuk dijadikan 'Diary' di database. Buat ringkasan yang komprehensif (3-4 kalimat), berbahasa Indonesia, menangkap emosi utama user, masalah yang dibahas, dan dukungan yang telah diberikan."
    user_prompt = f"Nama: {user_name}\nRiwayat ringkasan lama: {session_summary or 'N/A'}\n\nTeks percakapan lengkap sesi ini:\n{session_raw}"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content or "Ringkasan tidak tersedia."
    except Exception:
        return "Ringkasan tidak tersedia."
