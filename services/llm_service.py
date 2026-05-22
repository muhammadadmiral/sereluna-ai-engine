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
        model = os.getenv("NVIDIA_FAST_MODEL" if use_fast_model else "NVIDIA_MODEL", 
                          "meta/llama-3.1-8b-instruct" if use_fast_model else "meta/llama-3.3-70b-instruct")
        tiers.append(("NVIDIA NIM", lambda: _call_nvidia(model, messages, response_format, temperature, max_completion_tokens)))

    # Tier 2: Groq Fallbacks
    if provider_mode != "nvidia_only" and os.getenv("GROQ_API_KEY"):
        if use_fast_model:
            tiers.append(("Groq Fast", lambda: _call_groq(os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"), messages, response_format, temperature, max_completion_tokens)))
        else:
            tiers.append(("Groq Versatile", lambda: _call_groq(os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), messages, response_format, temperature, max_completion_tokens)))
            tiers.append(("Groq Kimi", lambda: _call_groq(os.getenv("GROQ_BACKUP_MODEL", "moonshotai/kimi-k2-instruct"), messages, response_format, temperature, max_completion_tokens)))

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
            print(f"✅ LLM Success: {name} ({elapsed:.2f}ms)")
            return res, name
        except Exception as e:
            last_error = e
            if _is_auth_or_billing_error(e):
                _PROVIDER_DISABLED_UNTIL[name] = time.time() + 900 # Cooldown 15 mins
            logger.warning("%s failed: %s", name, e)
            print(f"⚠️ {name} Error: {str(e)[:60]}... Falling back.")
            
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

def generate_dialog(
    user_message: str, screening_context: str, session_summary: str, profile_context: str,
    memory_context: str, recent_daily_context: str, risk_level: str, mood_signal: str,
    user_name: str, history_text: str, keywords: List[str], relevant_diary: Optional[str] = None,
    style_plan: Optional[Dict[str, Any]] = None, emotion_profile: Optional[Dict[str, Any]] = None,
    cognitive_distortions: Optional[Dict[str, Any]] = None, coping_pathway: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    safe_user_name = (user_name or "Teman").strip() or "Teman"
    style_plan_text = _format_style_plan(style_plan, safe_user_name)
    care_intel = _format_care_intelligence(emotion_profile, cognitive_distortions, coping_pathway)
    
    system_prompt = f"""Kamu adalah Sereluna, sahabat dekat {safe_user_name}. 
Bicara seperti teman nongkrong yang hangat, asyik, dan empati. GUNAKAN BAHASA INDONESIA KASUAL/GAUL.

IDENTITAS & GAYA:
- Panggil diri kamu 'Aku', jangan pernah pakai 'Saya'.
- Jangan kaku seperti asisten digital. Hilangkan kata-kata formal seperti 'Aktivitas', 'Aspek', 'Fisik'.
- Pakai gaya bahasa santai: 'ngobrol', 'cerita', 'santai aja', 'pasti berat ya'.
- Jika user pakai 'Gua/Lu', kamu wajib membalas dengan 'Gua/Lu' juga.

RESPONSE PLANNER:
{style_plan_text}

CARE INTELLIGENCE:
{care_intel}

KONTEKS:
- Mood: {mood_signal} | Risiko: {risk_level}
- Diary: {relevant_diary or "N/A"}

ATURAN MATI:
1. JANGAN pakai pembuka: "Saya senang", "Tentu saja", "Halo [Nama]".
2. Langsung masuk ke inti obrolan dengan nada akrab.
3. Maksimal 2-3 kalimat pendek kecuali diminta panjang.
4. Kirimkan JSON:
{{
  "reply": "balasan asik dan empati kamu",
  "session_summary": "ringkasan singkat",
  "sentiment_score": 3,
  "risk_flag": false
}}"""

    user_prompt = f"Riwayat Singkat: {session_summary}\n\nUser: {user_message}"

    try:
        # Force use the 'Strong' model (70B) for better personality
        content, _ = _completion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.85, # Higher temperature for more natural flow
            max_completion_tokens=completion_token_budget(style_plan),
            use_fast_model=False # Use 70B for higher quality
        )
        logger.info("Raw LLM Response: %s", content)
        parsed = _parse_json_object(content, {})
        
        # Validation: If reply is empty, try to use the raw content if it's not JSON
        reply = parsed.get("reply", "")
        if not reply and content and "{" not in content:
            reply = content
            
        reply = _polish_sereluna_reply(reply, safe_user_name, style_plan)
        
        return {
            "reply": reply or "Aku dengerin, ya. Bisa ceritain sedikit lagi?",
            "session_summary": clean_diary_summary(parsed.get("session_summary", "")),
            "sentiment_score": _coerce_score(parsed.get("sentiment_score"), 3),
            "suggested_action": parsed.get("suggested_action"),
            "risk_flag": bool(parsed.get("risk_flag", risk_level == "high")),
            "detected_symptoms": parsed.get("detected_symptoms", []),
            "dominant_category": parsed.get("dominant_category", "None")
        }
    except Exception as e:
        logger.error("Dialog generation failed: %s", e)
        return {"reply": "Aku dengerin, ya. Lanjutin aja ceritanya.", "session_summary": session_summary, "risk_flag": risk_level == "high"}

def generate_summary(session_raw: str, session_summary: str, user_name: str) -> str:
    system_prompt = "Buat ringkasan diary (3-4 kalimat) dalam Bahasa Indonesia. Langsung ke inti cerita, tanpa pembuka."
    user_prompt = f"Sesi: {session_raw}\n\nRolling Summary: {session_summary}"
    try:
        content, _ = _completion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.3, max_completion_tokens=500, use_fast_model=True
        )
        return clean_diary_summary(content)
    except Exception:
        return clean_diary_summary(session_summary or "Sesi percakapan selesai.")

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
    if not style: return "Gaya natural, santai."
    return f"Tone: {style.get('tone_guidance')}. Paragraf: {style.get('desired_paragraphs')}. Register: {style.get('user_register')}."

def _format_care_intelligence(emo: Any, cog: Any, cope: Any) -> str:
    primary = (emo or {}).get("primary_emotion", "netral")
    return f"Emosi: {primary}. Saran: {', '.join((cope or {}).get('steps', [])[:2])}."

def _polish_sereluna_reply(reply: str, name: str, style: Any) -> str:
    text = (reply or "").strip()
    # Strip repetitive openers
    text = re.sub(rf"^\s*(?:wah|aduh|duh|oke|baiklah|siap|tentu|{re.escape(name)})\s*[,!.]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:aku\s+(?:paham|ngerti|mengerti|dengerin))\s*[,!.]?\s*", "", text, flags=re.IGNORECASE)
    # Register fixes
    register = ((style or {}).get("user_register") or "").lower()
    if "gue-lu" in register:
        text = re.sub(r"\bAku\b", "Gua", text); text = re.sub(r"\baku\b", "gua", text)
        text = re.sub(r"\bKamu\b", "Lu", text); text = re.sub(r"\bkamu\b", "lu", text)
        text = re.sub(r"\bAnda\b", "Lu", text); text = re.sub(r"\banda\b", "lu", text)
    elif "saya-anda" not in register:
        text = re.sub(r"\bSaya\b", "Aku", text); text = re.sub(r"\bsaya\b", "aku", text)
        text = re.sub(r"\bAnda\b", "kamu", text); text = re.sub(r"\banda\b", "kamu", text)
    return text.strip().capitalize() if text else reply
