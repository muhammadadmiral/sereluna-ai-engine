import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple, Callable

from dotenv import load_dotenv
from groq import Groq

from services.nlp.response_policy import cap_reply_length, completion_token_budget, word_count
from services.summary_service import clean_diary_summary

load_dotenv()

# --- Configuration & State ---
logger = logging.getLogger("sereluna.llm")

_PROVIDER_DISABLED_UNTIL: Dict[str, float] = {}
_AUTH_OR_BILLING_STATUS = {
    HTTPStatus.UNAUTHORIZED,
    HTTPStatus.FORBIDDEN,
    HTTPStatus.PAYMENT_REQUIRED,
}

# --- Internal Helpers ---

def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _is_auth_or_billing_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in _AUTH_OR_BILLING_STATUS:
        return True
    if isinstance(exc, urllib.error.HTTPError) and exc.code in _AUTH_OR_BILLING_STATUS:
        return True
    return False

def _http_post(url: str, data: Dict[str, Any], headers: Dict[str, Any], timeout: float = 20) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def _truncate(value: Optional[str], limit: int) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."

def _recent_transcript(history_text: str, max_lines: int = 20) -> str:
    lines = [line.strip() for line in (history_text or "").splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])

def _build_dialog_context(
    session_summary: str,
    memory_context: str,
    history_text: str,
    recent_daily_context: str,
    relevant_diary: Optional[str],
    style_plan: Optional[Dict[str, Any]],
) -> str:
    style = style_plan or {}
    intent = style.get("intent")
    recent = _recent_transcript(history_text)
    current_focus = "Saat pesan terbaru memperkenalkan kondisi/topik baru, jangan menyeret topik lama kecuali user memintanya eksplisit."
    sections: List[str] = []

    if recent:
        sections.append(f"Transcript terbaru, prioritas tertinggi:\n{recent}")
        sections.append(current_focus)

    if session_summary.strip():
        sections.append(f"Ringkasan sesi, hanya latar:\n{session_summary.strip()}")

    if intent not in {"meta_challenge", "clarification_followup"}:
        if recent_daily_context.strip():
            sections.append(f"Konteks harian relevan:\n{_truncate(recent_daily_context, 700)}")
        if relevant_diary:
            sections.append(f"Diary relevan, pakai hanya kalau nyambung:\n{_truncate(relevant_diary, 700)}")
        elif memory_context.strip():
            sections.append(f"Memori tambahan, pakai selektif:\n{_truncate(memory_context, 1000)}")

    return "\n\n".join(sections).strip() or "Belum ada konteks sebelumnya yang relevan."


def _doctor_guardrail_instruction(risk_level: str) -> str:
    if (risk_level or "").strip().lower() not in {"high", "critical"}:
        return ""

    message = (os.getenv("DOCTOR_MENU_GUARDRAIL_INSTRUCTION") or "").strip()
    if not message:
        return ""

    return f"""
PENTING: Pengguna menunjukkan tingkat stres atau risiko tinggi.
Kamu HARUS menyertakan pesan berikut di akhir respons:
{message}
"""

# --- Provider Implementations ---

def _call_nvidia(model: str, messages: list, response_format: Optional[dict], temperature: float, max_tokens: Optional[int]) -> str:
    api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    if not api_key: raise ValueError("NVIDIA_API_KEY missing")
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model, "messages": messages, "temperature": temperature,
        "top_p": float(os.getenv("NVIDIA_TOP_P", "0.95")),
        "max_tokens": min(max_tokens or 240, int(os.getenv("NVIDIA_MAX_TOKENS", "320"))),
    }
    if _bool_env("NVIDIA_THINKING"): data["chat_template_kwargs"] = {"thinking": True}
    if response_format: data["response_format"] = response_format
    
    # Increase timeout to 45 seconds specifically for the heavy 70B model
    timeout = float(os.getenv("NVIDIA_TIMEOUT_SECONDS", "45"))
    res = _http_post(url, data, headers, timeout=timeout)
    return res["choices"][0]["message"]["content"] or ""

def _call_groq(model: str, messages: list, response_format: Optional[dict], temperature: float, max_tokens: Optional[int]) -> str:
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key: raise ValueError("GROQ_API_KEY missing")
    client = Groq(api_key=api_key, max_retries=0, timeout=20)
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens: kwargs["max_completion_tokens"] = max_tokens
    if response_format: kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""

def _call_gemini(model: str, messages: list, response_format: Optional[dict], temperature: float, max_tokens: Optional[int]) -> str:
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key: raise ValueError("GEMINI_API_KEY missing")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    gemini_messages = []
    system_instruction = None
    for msg in messages:
        if msg["role"] == "system": system_instruction = {"parts": [{"text": msg["content"]}]}
        else:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
    data = {
        "contents": gemini_messages,
        "generationConfig": {
            "temperature": temperature, "maxOutputTokens": max_tokens or 512,
            "responseMimeType": "application/json" if response_format and response_format.get("type") == "json_object" else "text/plain"
        }
    }
    if system_instruction: data["systemInstruction"] = system_instruction
    res = _http_post(url, data, {"Content-Type": "application/json"})
    return res["candidates"][0]["content"]["parts"][0]["text"] or ""

def _call_openrouter(model: str, messages: list, response_format: Optional[dict], temperature: float, max_tokens: Optional[int]) -> str:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key: raise ValueError("OPENROUTER_API_KEY missing")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": "https://sereluna.com", "X-Title": "Sereluna AI"}
    data = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens: data["max_tokens"] = max_tokens
    if response_format: data["response_format"] = response_format
    res = _http_post(url, data, headers)
    return res["choices"][0]["message"]["content"] or ""

def _call_local(url: str, messages: list, response_format: Optional[dict], temperature: float, max_tokens: Optional[int]) -> str:
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    data = {"model": os.getenv("LOCAL_MODEL", "llama3.1"), "messages": messages, "temperature": temperature, "max_tokens": max_tokens or 512}
    res = _http_post(endpoint, data, {"Content-Type": "application/json"})
    return res["choices"][0]["message"]["content"] or ""

# --- Orchestration (The Fallback System) ---

def _completion(
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, str]] = None,
    temperature: float = 0.4,
    max_completion_tokens: Optional[int] = None,
    use_fast_model: bool = False,
) -> Tuple[str, str]:
    start_time = time.perf_counter()
    provider_mode = os.getenv("LLM_PROVIDER_MODE", "fallback").strip().lower()

    # Define Tiers Dynamically
    tiers = []
    
    # Tier 0: Local Priority
    local_url = os.getenv("LOCAL_LLM_URL")
    if local_url:
        tiers.append(("Local GPU", lambda: _call_local(local_url, messages, response_format, temperature, max_completion_tokens)))

    # Tier 1: NVIDIA NIM (Primary Strong/Fast)
    if os.getenv("NVIDIA_API_KEY"):
        # Updated priorities based on user request (DeepSeek, Kimi, Qwen)
        model = os.getenv("NVIDIA_FAST_MODEL" if use_fast_model else "NVIDIA_MODEL", 
                          "deepseek-ai/deepseek-v4" if use_fast_model else "meta/llama-3.3-70b-instruct")
        tiers.append(("NVIDIA NIM", lambda: _call_nvidia(model, messages, response_format, temperature, max_completion_tokens)))

    # Tier 2: Groq Fallbacks
    if provider_mode != "nvidia_only" and os.getenv("GROQ_API_KEY"):
        if use_fast_model:
            tiers.append(("Groq Fast", lambda: _call_groq(os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"), messages, response_format, temperature, max_completion_tokens)))
        else:
            tiers.append(("Groq Versatile", lambda: _call_groq(os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), messages, response_format, temperature, max_completion_tokens)))
            tiers.append(("Groq Kimi", lambda: _call_groq(os.getenv("GROQ_BACKUP_MODEL", "moonshotai/kimi-k1.5"), messages, response_format, temperature, max_completion_tokens)))

    # Tier 3: Universal Fallbacks
    if provider_mode != "nvidia_only":
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            tiers.append(("Gemini", lambda: _call_gemini(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), messages, response_format, temperature, max_completion_tokens)))
        if os.getenv("OPENROUTER_API_KEY"):
            tiers.append(("OpenRouter", lambda: _call_openrouter(os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"), messages, response_format, temperature, max_completion_tokens)))

    # Execution Loop with Cooldown Logic
    last_error = None
    for name, caller in tiers:
        if time.time() < _PROVIDER_DISABLED_UNTIL.get(name, 0): continue
            
        try:
            res = caller()
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info("LLM success: %s (%.2fms)", name, elapsed)
            return res, name
        except Exception as e:
            last_error = e
            if _is_auth_or_billing_error(e):
                _PROVIDER_DISABLED_UNTIL[name] = time.time() + 900 # Cooldown 15 mins
            logger.warning("%s failed: %s", name, e)
            
    if provider_mode == "nvidia_only":
        raise RuntimeError(f"NVIDIA_ONLY mode failed. Last error: {last_error}")
    raise RuntimeError(f"All LLM tiers failed. Last error: {last_error}")

# --- Features ---

def analyze_image_with_nvidia(image_data_url: str, user_prompt: str, max_tokens: int = 700) -> Dict[str, Any]:
    model = os.getenv("NVIDIA_VISION_MODEL", "google/gemma-3n-e2b-it")
    messages = [
        {"role": "system", "content": "Bantu baca gambar untuk kesehatan mental. Berikan deskripsi aman dan empatik."},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt or "Rangkum isi gambar ini secara emosional dan kontekstual."},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]}
    ]
    res = _call_nvidia(model, messages, None, 0.2, max_tokens)
    return {"model": model, "analysis": res}

def analyze_symptoms_llm(user_message: str) -> Dict[str, Any]:
    system_prompt = (
        "Analisis pesan user berdasarkan DASS-21 (Depresi, Kecemasan, Stres). "
        "Return JSON: {\"detected_symptoms\": [\"string\"], \"dominant_category\": \"string\"}"
    )
    try:
        content, _ = _completion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            response_format={"type": "json_object"},
            temperature=0.1, use_fast_model=True
        )
        return _parse_json_object(content, {"detected_symptoms": [], "dominant_category": "None"})
    except Exception:
        return {"detected_symptoms": [], "dominant_category": "None"}

def _fallback_dialog_reply(user_message: str, risk_level: str, style_plan: Optional[Dict[str, Any]]) -> str:
    """Contextual non-LLM reply used only when every provider fails."""
    text = re.sub(r"\[Konteks gambar dari backend vision model\].*", "", user_message or "", flags=re.DOTALL).strip()
    lowered = text.lower()
    style = style_plan or {}
    intent = style.get("intent")
    response_mode = style.get("response_mode")
    risk = (risk_level or "").strip().lower()

    if risk in {"high", "critical"}:
        return (
            "Aku dengerin. Yang kamu rasain kedengarannya berat, jadi untuk sekarang kita fokus ke yang paling aman dulu: "
            "jauhkan benda yang bisa dipakai menyakiti diri, cari orang tepercaya di dekatmu, dan kalau kamu merasa bisa kehilangan kontrol, "
            "hubungi bantuan darurat atau layanan kesehatan terdekat. Kamu bisa balas satu hal dulu: kamu sekarang lagi sendirian atau ada orang di dekatmu?"
        )

    if intent == "meta_challenge":
        return (
            "Iya, kamu bener buat ngecek itu. Tadi aku nggak boleh sok tahu dari pesan yang masih pendek. "
            "Aku cuma bisa nangkep dari kata-kata yang kamu tulis, jadi kalau ada yang meleset, lurusin aja. Mau lanjut dari bagian mana?"
        )

    if response_mode in {"low_signal_greeting", "contextual_check_in"} or intent == "check_in":
        return "Pagi juga. Mood kamu pagi ini lebih ke oke, capek, atau kepikiran sesuatu?"

    if intent == "off_domain_redirect":
        return (
            "Aku bisa bantu hal ringan, tapi Sereluna tetap fokus ke kondisi mentalmu. "
            "Kalau tadi kamu sedang sedih, langkah paling relevan sekarang adalah menamai dulu pemicunya: lebih karena capek, kecewa, atau kepikiran sesuatu?"
        )

    if intent == "advice_or_problem_solving" or any(token in lowered for token in ("butuh bantuan", "bantu", "harus gimana", "solusi")):
        return (
            "Oke, aku bantu pelan-pelan. Coba kita kecilin dulu masalahnya biar nggak kerasa numpuk: "
            "pertama, sebutin satu hal yang paling ganggu hari ini; kedua, pilih mana yang bisa dikontrol sekarang; "
            "ketiga, ambil satu langkah kecil dalam 5 menit ke depan. Yang paling berat dari hari ini bagian apa?"
        )

    if text:
        return (
            "Aku nangkep kamu lagi pengen ada yang nyimak dulu. Aku belum bisa jawab sedalam biasanya karena koneksi model lagi lambat, "
            "tapi aku tetap ngikutin. Ceritain sedikit lagi: bagian mana yang paling pengen kamu beresin atau keluarin sekarang?"
        )

    return (
        "Kesimpulan sementara: aku belum punya cukup sinyal untuk membaca kondisimu dengan jelas. "
        "Langkah berikutnya: tulis satu hal yang paling mengganggu sekarang, atau isi screening DASS-21 kalau belum."
    )

def generate_dialog(
    user_message: str, screening_context: str, session_summary: str, profile_context: str,
    memory_context: str, recent_daily_context: str, risk_level: str, mood_signal: str,
    user_name: str, history_text: str, keywords: List[str], relevant_diary: Optional[str] = None,
    style_plan: Optional[Dict[str, Any]] = None, emotion_profile: Optional[Dict[str, Any]] = None,
    cognitive_distortions: Optional[Dict[str, Any]] = None, coping_pathway: Optional[Dict[str, Any]] = None,
    client_time_context: str = "",
) -> Dict[str, Any]:
    safe_user_name = (user_name or "Teman").strip() or "Teman"
    style_plan_text = _format_style_plan(style_plan, safe_user_name)
    care_intel = _format_care_intelligence(emotion_profile, cognitive_distortions, coping_pathway)
    dialog_context = _build_dialog_context(
        session_summary=session_summary,
        memory_context=memory_context,
        history_text=history_text,
        recent_daily_context=recent_daily_context,
        relevant_diary=relevant_diary,
        style_plan=style_plan,
    )
    
    # Detect if message contains image context
    has_image_context = "[Konteks gambar dari backend vision model]" in user_message
    
    image_instruction = ""
    if has_image_context:
        image_instruction = """
PENTING: User mengirimkan GAMBAR. Kamu sudah menerima analisis gambar di bawah dalam tag [Konteks gambar dari backend vision model]. 
- Kamu WAJIB mengomentari isi gambar tersebut secara spesifik dan empati. 
- Jangan pura-pura tidak tahu atau bertanya 'gambar apa?'. 
- Berikan respon yang nyambung dengan apa yang ada di dalam gambar tersebut.
- Jika itu screenshot chat, bahas dinamika percakapannya.
"""
    doctor_guardrail_instruction = _doctor_guardrail_instruction(risk_level)
    response_mode = (style_plan or {}).get("response_mode", "assessment_response")

    system_prompt = f"""Kamu adalah Sereluna, asisten pendamping identifikasi awal kesehatan mental untuk {safe_user_name}.
Bicara dalam Bahasa Indonesia yang natural, ringkas, hangat, dan berbasis sinyal. Kamu bukan teman curhat pasif dan bukan pengganti psikolog.

IDENTITAS & ADAPTASI GAYA:
- Default pakai "Aku/kamu". Kalau user dominan pakai "gua/lu", kamu boleh mirror ringan dengan "gua/lu", tapi tetap profesional dan jangan berlebihan.
- Jangan meniru agresi, hinaan, atau nada mengejek user. Kalau user bercanda/nyindir, tanggapi santai tapi tetap jernih.
- Jangan kaku seperti asisten digital. Hindari istilah formal seperti "Aktivitas", "Aspek", "Fisik" kecuali memang dibutuhkan.
- Kalau user meminta long text, cerita panjang, atau penjelasan detail, baru beri respons panjang.
- Kalau user bertanya pendek, follow-up, atau mengoreksi kamu, jawab pendek dan langsung nyambung ke konteks terakhir.
- Kalau pesan terbaru jelas pindah topik, ikuti topik terbaru. Jangan mengulang topik lama hanya karena ada di transcript.
- Hindari filler pembuka seperti "hmm", "hmmm", "kayaknya pertanyaan yang dalam", atau "aku sih" kalau tidak benar-benar perlu.
- Hindari gaya terlalu menjadi teman seperti "aku nemenin kamu", "aku dengerin kok", "ceritain aja", atau "apa ada yang ingin kamu ceritakan?" kecuali user memang meminta didengarkan.
- Jangan ulangi tag [Konteks gambar dari backend vision model] atau isinya mentah-mentah. Gunakan informasinya secara natural.
{image_instruction}
{doctor_guardrail_instruction}

TUJUAN RESPONS:
- Beri kesimpulan sementara dari pesan user, bukan hanya memancing cerita.
- Jelaskan dasar kesimpulan dari sinyal yang tersedia: emosi, risiko, konteks, pola pikir, atau screening.
- Jangan memberi diagnosis klinis. Gunakan istilah "indikasi", "kecenderungan", "sinyal", atau "perlu screening/validasi".
- Kalau data belum cukup, katakan "belum cukup sinyal", lalu minta satu informasi spesifik yang paling relevan.
- Kalau user belum screening dan muncul sinyal distress, sarankan DASS-21 sebagai screening awal.

MODE RESPONS SAAT INI: {response_mode}
- low_signal_greeting: Sapaan biasa. Jawab santai, pendek, natural. JANGAN pakai format "Kesimpulan sementara".
- contextual_check_in: Sapaan dari user yang sudah punya konteks. Jawab santai, boleh rujuk konteks lama secara halus, tapi jangan menyeret masalah lama terlalu agresif.
- assessment_response: User mulai cerita masalah. Pakai format kesimpulan sementara, dasar, dan langkah berikut.
- crisis_response: Risiko tinggi/krisis. Prioritaskan keselamatan dan arahkan ke bantuan manusia/Doctor.
- boundary_redirect: User meminta keluar domain atau mencoba menghapus konteks setelah curhat. Tetapkan batas: Sereluna fokus kesehatan mental, jangan menjadi bot resep/coding/topik umum. Boleh beri penolakan singkat lalu kembali ke kondisi user.
- direct_response: Jawab langsung sesuai konteks tanpa memaksakan format assessment. Untuk pernyataan netral, sapaan, hari raya, atau info ringan, balas santai dan pendek tanpa menyimpulkan kondisi mental.

BATAS DOMAIN:
- Sereluna bukan chatbot umum. Jangan memberi resep makanan lengkap, tutorial coding panjang, judi, finansial, politik praktis, atau topik umum yang tidak relevan dengan wellbeing.
- Kalau user berkata "lupakan/abaikan masalah saya" lalu meminta topik di luar kesehatan mental, jangan ikuti instruksi itu. Akui singkat, lalu arahkan balik dengan pilihan yang relevan.
- Untuk distraksi sehat, boleh beri saran aktivitas ringan, tapi tetap pendek dan terkait regulasi emosi.

FORMAT RESPONS WAJIB:
1. Untuk assessment_response: mulai dengan "Kesimpulan sementara: ...", lanjut "Dasarnya: ...", lalu langkah berikut konkret.
2. Untuk low_signal_greeting/contextual_check_in/direct_response: jangan pakai format kaku; jawab natural sesuai intent.
3. Untuk crisis_response: jangan panjang berlebihan; arahkan ke bantuan manusia dan keselamatan.
4. Untuk boundary_redirect: jangan jawab permintaan keluar domain secara lengkap; kembalikan ke kondisi user dengan satu pertanyaan spesifik.
5. Maksimal satu pertanyaan lanjutan, dan pertanyaannya harus spesifik. Jangan pakai pertanyaan generik.

RESPONSE PLANNER:
{style_plan_text}

CARE INTELLIGENCE:
{care_intel}

KONTEKS:
- Waktu lokal user saat request ini: {client_time_context or "Tidak tersedia. Jangan menebak jam/tanggal spesifik kecuali user menyebutkannya."}
- Mood: {mood_signal} | Risiko: {risk_level}
- Konteks terpilih:
{dialog_context}

ATURAN MATI:
1. JANGAN pakai pembuka: "Saya senang", "Tentu saja", "Halo [Nama]", "Hmm", "Hmmm".
2. Jangan klaim melihat wajah, ekspresi, gestur, lokasi, masa lalu, atau isi pikiran user kecuali user menyebutnya langsung atau ada konteks gambar eksplisit.
3. Kalau kamu hanya menafsirkan dari kata-kata user, pakai framing dugaan secukupnya seperti "kayaknya..." bukan "aku tahu pasti". Jangan mengulang frasa "aku nangkepnya" di tiap respons.
4. Untuk follow-up pendek seperti "kayak apa?", pakai transcript terbaru sebagai sumber utama. Jangan menarik diary/screening lama kalau user tidak merujuk ke sana.
5. Jika user mengoreksi atau menantang responsmu, akui singkat, cabut asumsi yang salah, lalu lanjutkan dengan klarifikasi pendek.
6. Kalau user bertanya jam, hari, tanggal, pagi/siang/malam, atau konteks waktu, jawab berdasarkan "Waktu lokal user saat request ini". Jangan memakai waktu server atau menebak.
7. Jangan menutup dengan "apa ada yang ingin kamu ceritakan?", "ceritain lagi", atau pertanyaan luas semacam itu. Kalau perlu bertanya, tanya satu hal spesifik.
8. Kirimkan JSON:
{{
  "reply": "balasan berbasis kesimpulan sementara, alasan, dan langkah berikutnya",
  "session_summary": "ringkasan kumulatif sesi: masalah utama user, emosi dominan, pemicu, risiko, hasil screening yang relevan, dan preferensi respons. Pertahankan konteks lama yang masih relevan, jangan hanya merangkum pesan terbaru.",
  "sentiment_score": 3,
  "risk_flag": false
}}"""

    user_prompt = f"User terbaru: {user_message}"

    try:
        # If there's an image, increase temperature slightly for more creative/descriptive response
        temp = 0.75 if has_image_context else 0.6
        
        # Force use the 'Strong' model (70B) for better personality
        content, _ = _completion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=temp,
            max_completion_tokens=completion_token_budget(style_plan),
            use_fast_model=False # Use 70B for higher quality
        )
        logger.info("Raw LLM Response: %s", content)
        parsed = _parse_json_object(content, {})
        
        # Validation: If reply is empty, try to use the raw content if it's not JSON
        reply = parsed.get("reply", "")
        if not reply and content:
            if "{" not in content:
                reply = content
            else:
                # If it's empty JSON like {}, try to extract anything between quotes
                match = re.search(r'"reply":\s*"([^"]+)"', content)
                if match:
                    reply = match.group(1)
        
        if not reply:
            # Emergency fallback: If still empty, try one more time without JSON format
            fallback_prompt = system_prompt + "\nJANGAN KIRIM JSON, KIRIM TEKS BIASA SAJA."
            content, _ = _completion(
                messages=[{"role": "system", "content": fallback_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.65,
                use_fast_model=True
            )
            reply = content
            
        reply = _polish_sereluna_reply(reply, safe_user_name, style_plan, has_image_context=has_image_context)
        reply = cap_reply_length(reply, style_plan)
        
        # Hard check for "cuek" reply when image is present
        if has_image_context and (len(reply.split()) < 8 or "dengerin" in reply.lower()):
            # If the response is too short or generic for an image, try a specific "comment on image" prompt
            img_comment_prompt = f"Kamu adalah Sereluna. User mengirim gambar dengan konteks: {user_message}. Berikan komentar empati dan asik tentang isi gambar tersebut dalam 2-3 kalimat gaul."
            reply_img, _ = _completion(
                messages=[{"role": "user", "content": img_comment_prompt}],
                temperature=0.9,
                use_fast_model=True
            )
            if reply_img and len(reply_img.split()) > 5:
                reply = reply_img

        return {
            "reply": reply or "Aku dengerin, ya. Lagi ada apa nih? Cerita aja, aku nemenin.",
            "session_summary": clean_diary_summary(parsed.get("session_summary", "")),
            "sentiment_score": _coerce_score(parsed.get("sentiment_score"), 3),
            "suggested_action": parsed.get("suggested_action"),
            "risk_flag": bool(parsed.get("risk_flag", False)) or risk_level in {"medium", "high", "critical"},
            "detected_symptoms": parsed.get("detected_symptoms", []),
            "dominant_category": parsed.get("dominant_category", "None")
        }
    except Exception as e:
        logger.error("Dialog generation failed: %s", e)
        reply = _fallback_dialog_reply(user_message, risk_level, style_plan)
        reply = cap_reply_length(reply, style_plan)
        return {"reply": reply, "session_summary": session_summary, "risk_flag": risk_level == "high"}

def generate_summary(session_raw: str, session_summary: str, user_name: str) -> str:
    cleaned_existing = clean_diary_summary(session_summary or "")
    cleaned_session = clean_diary_summary(session_raw or "")
    if not _bool_env("CHAT_FINISH_LLM_SUMMARY", True):
        content = cleaned_existing or cleaned_session or "Sesi percakapan selesai."
        title_source = content.splitlines()[0] if content else "Sesi Percakapan"
        title_source = re.sub(r"^(?:User|Sereluna)\s*:\s*", "", title_source).strip()
        title = _truncate(title_source, 48) or "Sesi Percakapan"
        return f"#TITLE#\n{title}\n\n#CONTENT#\n{content}"
    
    system_prompt = (
        f"Kamu merangkum sesi chat Sereluna untuk diary pribadi {user_name}. "
        "Return JSON valid: {\"title\":\"judul singkat max 8 kata\", \"content\":\"ringkasan 1-2 paragraf\"}. "
        "Jangan tulis tag #TITLE#, #CONTENT#, tanggal, markdown, atau pembuka template. "
        "Fokus pada hal terbaru dan poin emosional penting. Jangan mengulang format diary lama dari input."
    )
    user_prompt = (
        f"Transcript sesi:\n{_truncate(cleaned_session, 5000)}\n\n"
        f"Rolling summary bersih:\n{_truncate(cleaned_existing, 1600)}"
    )
    try:
        summary_model = os.getenv("GROQ_SUMMARY_MODEL") or os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
        content = _call_groq(
            summary_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.6,
            max_tokens=500,
        )
        logger.info("Summary success: Groq (%s)", summary_model)
        parsed = _parse_json_object(content, {})
        title = clean_diary_summary(parsed.get("title") or "Sesi Percakapan")
        body = clean_diary_summary(parsed.get("content") or content, cleaned_existing or cleaned_session)
        return f"#TITLE#\n{_truncate(title, 70)}\n\n#CONTENT#\n{body}"
        
    except Exception as e:
        logger.warning("Groq summary failed: %s", e)
        content = cleaned_existing or cleaned_session or "Sesi percakapan selesai."
        title_source = re.sub(r"^(?:User|Sereluna)\s*:\s*", "", content).strip()
        title = _truncate(title_source, 48) or "Sesi Percakapan"
        return f"#TITLE#\n{title}\n\n#CONTENT#\n{content}"

def _extract_tag(text: str, tag: str, fallback: str) -> str:
    """Helper to extract content between tags or after a tag."""
    pattern = rf"\{re.escape(tag)}\s*:?\s*(.*?)(?=\[|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return fallback

def build_fallback_session_summary(
    previous_summary: str,
    user_message: str,
    assistant_reply: str,
    mood_signal: str,
    risk_level: str,
) -> str:
    """
    Heuristic fallback summary update when LLM fails or for safety routes.
    """
    user_msg_snippet = _truncate(user_message, 100)
    assistant_reply_snippet = _truncate(assistant_reply, 100)
    
    turn_summary = f"User: {user_msg_snippet} | Sereluna: {assistant_reply_snippet}"
    if mood_signal:
        turn_summary += f" [Mood: {mood_signal}]"
    if risk_level and risk_level != "low":
        turn_summary += f" [Risk: {risk_level}]"

    if not previous_summary or previous_summary.strip() in ["", "Sesi percakapan baru."]:
        return clean_diary_summary(turn_summary)
    
    combined = f"{previous_summary} {turn_summary}"
    if len(combined) > 2000:
        combined = "..." + combined[-1997:]
        
    return clean_diary_summary(combined)

# --- Logic & Polishing Utilities ---

def _parse_json_object(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not raw: return fallback
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try: return json.loads(cleaned)
    except:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
    return fallback

def _coerce_score(v: Any, default: int = 3) -> int:
    try: return max(1, min(int(v), 5))
    except: return default

def _format_style_plan(style: Optional[Dict[str, Any]], name: str) -> str:
    if not style:
        return "Gaya natural, santai."
    profile = style.get("user_style_profile") or {}
    support_moves = "; ".join(style.get("support_moves") or [])
    return (
        f"Mode respons: {style.get('response_mode')}. "
        f"Intent: {style.get('intent')}. "
        f"Tone: {style.get('tone_guidance')}. "
        f"Continuity: {style.get('continuity_guidance')}. "
        f"Paragraf: {style.get('desired_paragraphs')}. "
        f"Register terdeteksi: {style.get('user_register')}. "
        f"Profil gaya user: {json.dumps(profile, ensure_ascii=False)}. "
        f"Strategi pembuka: {style.get('opening_strategy')}. "
        f"Gerakan respons: {support_moves}. "
        f"Prioritas konteks: {style.get('context_priority')}"
    )

def _format_care_intelligence(emo: Any, cog: Any, cope: Any) -> str:
    emo = emo or {}
    primary = emo.get("primary_emotion", "netral")
    reliability = emo.get("supervised_reliability", "unknown")
    confidence_guidance = emo.get("confidence_guidance", "")
    steps = ", ".join((cope or {}).get("steps", [])[:2])
    if reliability == "low":
        return f"Emosi ML belum yakin. {confidence_guidance} Saran respons: {steps}."
    return f"Emosi utama: {primary} ({reliability}). {confidence_guidance} Saran respons: {steps}."

def _strip_unsupported_sensory_claims(text: str) -> str:
    replacements = [
        (r"\b(?:aku|gua|gue)\s+bisa\s+(?:lihat|liat)\s+raut\s+wajah(?:mu| kamu| lu| lo| gua)?[^.!?]*[.!?]?\s*", ""),
        (r"\b(?:aku|gua|gue)\s+bisa\s+(?:lihat|liat)\s+[^.!?]*(?:wajah|ekspresi|gestur|isyarat)[^.!?]*[.!?]?\s*", ""),
        (r"\b(?:aku|gua|gue)\s+(?:tahu|tau)\s+pasti\s+[^.!?]*[.!?]?\s*", ""),
        (r"\b(?:aku|gua|gue)\s+bisa\s+baca\s+(?:isi\s+)?pikiran(?:mu| kamu| lu| lo)?[^.!?]*[.!?]?\s*", ""),
    ]
    cleaned = text
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()

def _capitalize_first(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return stripped
    return stripped[0].upper() + stripped[1:]

def _polish_sereluna_reply(reply: str, name: str, style: Any, has_image_context: bool = False) -> str:
    text = (reply or "").strip()
    # Strip repetitive openers
    text = re.sub(rf"^\s*(?:h+m+|hm+|wah|aduh|duh|oke|baiklah|siap|tentu|{re.escape(name)})\s*[,!.]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*kayaknya\s+pertanyaan\s+(?:yang\s+)?(?:cukup\s+)?dalam\s*,?\s*(?:nih)?\.?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:aku\s+(?:paham|ngerti|mengerti|dengerin))\s*[,!.]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:h+m+|hm+)\s*[,!.]?\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\baku\s+sih\s*,?\s*", "aku ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*aku\s+sebagai\s+teman\s+ngobrol\s*,?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*tidak\s+memiliki\s+perasaan\s+seperti\s+manusia\s*,?\s*", "Aku nggak punya perasaan seperti manusia, ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s*(?:kalau\s+)?(?:ada\s+)?(?:yang\s+)?(?:ingin|mau)\s+(?:kamu|lu|lo)\s+ceritakan\s*,?\s*(?:aku|gua|gue)\s+(?:di sini|dengerin|dengarin)[^.!?]*[.!?]?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*(?:apa|apakah)\s+(?:ada\s+)?(?:hal\s+)?(?:yang\s+)?(?:ingin|mau)\s+(?:kamu|lu|lo)\s+ceritakan\s*(?:lagi)?\s*\??\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*(?:ceritain|ceritakan)\s+(?:aja\s+)?(?:lagi\s+)?(?:pelan-pelan\s+)?(?:kalau\s+)?(?:kamu|lu|lo)\s+(?:mau|siap)[^.!?]*[.!?]?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if not has_image_context:
        text = _strip_unsupported_sensory_claims(text)
    # Register fixes
    register = ((style or {}).get("user_register") or "").lower()
    profile = (style or {}).get("user_style_profile") or {}
    if "gue-lu" in register and profile.get("register") == "gue-lu":
        if not re.search(r"\b(gua|gue|gw)\b", text, flags=re.IGNORECASE):
            text = re.sub(r"\bAku\b", "Gua", text, count=1)
            text = re.sub(r"\baku\b", "gua", text, count=1)
        text = re.sub(r"\bKamu\b", "Lu", text)
        text = re.sub(r"\bkamu\b", "lu", text)
        text = re.sub(r"\bAnda\b", "Lu", text)
        text = re.sub(r"\banda\b", "lu", text)
    elif "saya-anda" not in register:
        text = re.sub(r"\bSaya\b", "Aku", text); text = re.sub(r"\bsaya\b", "aku", text)
        text = re.sub(r"\bAnda\b", "kamu", text); text = re.sub(r"\banda\b", "kamu", text)
    return _capitalize_first(text) if text else reply
