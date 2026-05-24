import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from schemas.chat_schema import ChatFinishRequest, ChatRequest, ChatResponse, ClinicalInsight, UIMetadata
from services.context_service import (
    ensure_user_document,
    finish_session,
    format_messages,
    get_chat_context,
    get_or_create_session,
    get_or_create_today_diary,
    get_session,
    get_session_messages,
    mark_session_finishing,
    save_message,
    update_chat_summaries,
)
from services.firebase_service import get_current_user
from services.llm_service import build_fallback_session_summary, generate_dialog, generate_summary
from services.media_service import analyze_images_for_chat
from services.nlp_service import build_context_algorithm_result, build_response_style_plan
from services.notification_service import create_notification
from services.nlp.utils import is_greeting_only
from services.summary_service import clean_diary_summary

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
logger = logging.getLogger("sereluna.chat")
logger.setLevel(logging.INFO)
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Jakarta")

# Ensure logs are visible in the console
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:     %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False


def _log_chat_pipeline(trace_id: str, event: str, payload: Dict[str, Any]) -> None:
    logger.info(
        "sereluna_chat_pipeline %s",
        json.dumps({"trace_id": trace_id, "event": event, **payload}, ensure_ascii=False, default=str),
    )


def _log_professor_demo(user_text: str, reply: str, algorithm_result: Dict[str, Any]) -> None:
    try:
        eval_data = algorithm_result.get("supervised_model_evaluation", {})
        accuracy = eval_data.get("accuracy", "N/A")
        macro_f1 = eval_data.get("macro_f1", "N/A")

        preprocessing_filter = algorithm_result.get("preprocessing_filter", {})
        filter_method = (preprocessing_filter.get("algorithm", {})).get("method", "N/A")
        normalized = preprocessing_filter.get("normalized_text", "N/A")

        supervised = algorithm_result.get("supervised_emotion_classifier") or {}
        intent_data = algorithm_result.get("intent_classifier") or {}
        predicted_emotion = supervised.get("predicted_emotion", "N/A")
        confidence = supervised.get("confidence", 0.0)
        top_probs = supervised.get("top_probabilities", [])
        prob_text = ", ".join([f"{p['emotion']}: {p['probability']}" for p in top_probs])
        
        # XAI - Explainable AI Features
        xai_features = supervised.get("explainable_features", [])
        if xai_features and "error" in xai_features[0]:
            xai_text = xai_features[0]["error"]
        else:
            xai_text = ", ".join([f"'{f.get('feature', 'unknown')}': {f.get('impact', 0)}" for f in xai_features])

        risk_level = algorithm_result.get("risk_level", "unknown")
        distortions = algorithm_result.get("cognitive_distortions", {})
        distortion_list = distortions.get("detected_patterns", [])

        log_msg = "\n" + "="*75 + "\n"
        log_msg += "SERELUNA AI - BACKEND INFERENCE TRACE\n"
        log_msg += "="*75 + "\n"
        log_msg += f"1. DATA MINING & XAI (EXPLAINABLE INFERENCE)\n"
        log_msg += f"   - Input Teks       : {user_text}\n"
        log_msg += f"   - Hasil Normalisasi: {normalized}\n"
        log_msg += f"   - Pemicu Emosi (XAI): {xai_text if xai_features else 'None (Too short)'}\n"
        log_msg += f"   - Distorsi Kognitif : {', '.join(distortion_list) if distortion_list else 'None'}\n"
        log_msg += "-"*75 + "\n"
        log_msg += f"2. MACHINE LEARNING (SUPERVISED EMOTION MODEL)\n"
        log_msg += f"   - Model Performance : Accuracy {accuracy} | Macro-F1 {macro_f1}\n"
        log_msg += f"   - Prediksi Emosi    : {predicted_emotion.upper()} (Conf: {confidence})\n"
        log_msg += f"   - Top Probabilities : {prob_text}\n"
        log_msg += "-"*75 + "\n"
        log_msg += f"3. HYBRID ROUTING & SENTIMENT TREND\n"
        log_msg += f"   - Trend Sentiment   : {algorithm_result.get('sentiment_trend', 'Stable')}\n"
        log_msg += f"   - Routing Logic     : {algorithm_result.get('routing_mode', 'Direct ML')}\n"
        log_msg += f"   - Intent Classifier : {intent_data.get('intent', 'N/A')} (Conf: {intent_data.get('confidence', 'N/A')})\n"
        log_msg += "-"*75 + "\n"
        log_msg += f"4. FINAL LLM RESPONSE (SENT TO USER)\n"
        log_msg += f"   - Output Bot        : {reply}\n"
        log_msg += "="*75 + "\n"
        
        logger.info(log_msg)
    except Exception as e:
        logger.error(f"Failed to print professor log: %s", e)


def _debug_metadata(algorithm_result: Dict[str, Any], routed_to: str, elapsed_ms: int) -> Dict[str, Any]:
    supervised = algorithm_result.get("supervised_emotion_classifier") or {}
    evaluation = algorithm_result.get("supervised_model_evaluation") or {}
    emotion_profile = algorithm_result.get("emotion_profile") or {}
    return {
        "processed_by_backend_first": True,
        "routed_to": routed_to,
        "elapsed_ms": elapsed_ms,
        "steps": [
            "save_user_message",
            "load_chat_context",
            "preprocessing_filter",
            "risk_classification",
            "sentiment_scoring",
            "emotion_classification",
            "cognitive_distortion_detection",
            "coping_pathway_planning",
            "response_style_planning",
            routed_to,
        ],
        "risk_level": algorithm_result.get("risk_level"),
        "sentiment_score": algorithm_result.get("sentiment_score"),
        "primary_emotion": emotion_profile.get("primary_emotion"),
        "supervised_emotion": supervised.get("predicted_emotion"),
        "supervised_confidence": supervised.get("confidence"),
        "model_accuracy": evaluation.get("accuracy"),
        "model_macro_f1": evaluation.get("macro_f1"),
        "dataset_size": evaluation.get("dataset_size"),
    }


def _safety_reply() -> str:
    doctor_message = (os.getenv("DOCTOR_MENU_GUARDRAIL_INSTRUCTION") or "").strip()
    if doctor_message:
        return (
            "Aku sangat peduli sama kamu, dan sepertinya kondisimu sedang sangat berat. "
            f"{doctor_message} Aku juga bakal tetap nemenin kamu di sini."
        )

    return (
        "Aku sangat peduli sama kamu, dan sepertinya kondisimu sedang sangat berat. "
        "Kamu nggak sendirian ya. Sereluna punya fitur Konselor di mana kamu bisa ngobrol langsung dengan ahlinya. "
        "Coba cek menu Konselor ya, ada yang siap dengerin kamu. Aku juga bakal tetap nemenin kamu di sini."
    )


def _background_save_and_update(
    uid: str,
    room_id: str,
    session_id: str,
    reply: str,
    analysis_results: Dict[str, Any],
    next_summary: str,
):
    save_message(
        uid=uid,
        diary_id=room_id,
        session_id=session_id,
        role="assistant",
        text=reply,
        analysis_results=analysis_results,
    )
    update_chat_summaries(uid, room_id, session_id, next_summary)


def _format_media_context(media_results: list[Dict[str, Any]]) -> str:
    lines = []
    for index, item in enumerate(media_results, start=1):
        if item.get("analysis"):
            lines.append(f"Gambar {index}: {item['analysis']}")
        elif item.get("error"):
            lines.append(f"Gambar {index}: gagal dianalisis ({item['error']}).")
    return "\n".join(lines)


def _is_doctor_consultation_request(text: str) -> bool:
    return bool(re.search(r"\b(?:psikolog|dokter|konsul|konsultasi|whatsapp|wa)\b", text or "", re.IGNORECASE))


def _doctor_direct_reply() -> str:
    return (os.getenv("DOCTOR_DIRECT_REPLY") or "").strip()


def _client_time_context(request: ChatRequest) -> Dict[str, str]:
    timezone_name = (request.client_timezone or "").strip()
    utc_offset = (request.client_utc_offset or "").strip()
    local_datetime = (request.client_local_datetime or "").strip()

    if not local_datetime:
        zone_name = timezone_name or APP_TIMEZONE
        try:
            now = datetime.now(ZoneInfo(zone_name))
            local_datetime = now.isoformat(timespec="seconds")
            timezone_name = timezone_name or zone_name
            utc_offset = utc_offset or now.strftime("%z")
            if len(utc_offset) == 5:
                utc_offset = f"{utc_offset[:3]}:{utc_offset[3:]}"
        except Exception:
            now = datetime.now()
            local_datetime = now.isoformat(timespec="seconds")
            timezone_name = timezone_name or APP_TIMEZONE

    label_parts = [local_datetime]
    if timezone_name:
        label_parts.append(timezone_name)
    if utc_offset:
        label_parts.append(f"UTC{utc_offset}" if utc_offset.startswith(("+", "-")) else utc_offset)

    return {
        "client_timezone": timezone_name,
        "client_utc_offset": utc_offset,
        "client_local_datetime": local_datetime,
        "prompt": " | ".join(label_parts),
    }


@router.post("/", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    trace_id = uuid4().hex[:12]
    started_at = time.perf_counter()
    user_text = (request.text or "").strip()
    media_ids = [str(media_id).strip() for media_id in (request.media_ids or []) if str(media_id).strip()]
    
    if not user_text and not media_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="text is required")
    if not user_text and media_ids:
        user_text = "Tolong bantu baca gambar yang aku kirim."
    time_context = _client_time_context(request)

    uid = current_user["uid"]
    _log_chat_pipeline(trace_id, "request_received", {
        "uid": uid, 
        "message_length": len(user_text),
        "media_count": len(media_ids)
    })
    
    ensure_user_document(uid, current_user)
    room_id, _diary = get_or_create_today_diary(uid, request.room_id)
    session_id, _session = get_or_create_session(uid, room_id, request.session_id)

    # 1. Media Analysis (Vision Model)
    media_results = analyze_images_for_chat(uid, media_ids) if media_ids else []
    media_context = _format_media_context(media_results)
    llm_user_text = user_text
    if media_context:
        llm_user_text = f"{user_text}\n\n[Konteks gambar dari backend vision model]\n{media_context}"

    # 2. Save User Message
    save_message(
        uid=uid,
        diary_id=room_id,
        session_id=session_id,
        role="user",
        text=user_text,
        analysis_results={
            "mood_signal": request.mood_signal or "",
            "has_image": bool(media_ids),
            "media_ids": media_ids,
            "media_analysis": media_results,
            "client_time": time_context,
        },
    )

    # 3. Load Context & Run Backend Algorithms
    context = get_chat_context(uid, room_id, session_id)
    history_text = context["session_history"]
    
    algorithm_result = build_context_algorithm_result(
        text=llm_user_text,
        mood_signal=request.mood_signal or "",
        screening_context=context["latest_screening_summary"],
        session_summary=context["session_summary"],
        past_diaries=context["past_diaries"],
    )
    
    # Sentiment trend tracker
    recent_sentiments = []
    for line in history_text.splitlines()[-10:]:
        if "Sentiment:" in line:
            try: recent_sentiments.append(int(line.split("Sentiment:")[1].strip()))
            except: pass
    
    avg_sentiment = sum(recent_sentiments) / len(recent_sentiments) if recent_sentiments else 3
    current_sentiment = algorithm_result["sentiment_score"]
    
    trend = "Stable"
    if current_sentiment < avg_sentiment - 0.5: trend = "Declining (User getting worse)"
    elif current_sentiment > avg_sentiment + 0.5: trend = "Improving (User feeling better)"
    algorithm_result["sentiment_trend"] = trend

    # Uncertainty-aware routing
    supervised = algorithm_result.get("supervised_emotion_classifier") or {}
    confidence = supervised.get("confidence", 0.0)
    risk_level = algorithm_result["risk_level"]
    
    routing_mode = "Direct ML Inference"
    if confidence < 0.40 and not is_greeting_only(user_text):
        routing_mode = "Hybrid LLM Validation (ML Uncertainty Detected)"
    algorithm_result["routing_mode"] = routing_mode

    # 4. Response Planning
    style_plan = build_response_style_plan(
        text=llm_user_text,
        mood_signal=request.mood_signal or "",
        risk_level=risk_level,
        sentiment_score=current_sentiment,
        session_summary=context["session_summary"],
        history_text=history_text,
    )
    
    if trend == "Declining (User getting worse)":
        style_plan["tone_guidance"] += "; USE DEEP EMPATHY MODE: be extremely gentle, validate deeply."

    # 5. Safety Route Logic (Crisis/Toxicity Detection)
    risk_trace = algorithm_result.get("risk", {})
    route_to_safety = risk_trace.get("reason") in {"current_crisis_signal", "preprocessing_crisis_filter"}
    
    if route_to_safety:
        reply = _safety_reply()
        next_summary = build_fallback_session_summary(context["session_summary"], llm_user_text, reply, request.mood_signal or "", risk_level)
        background_tasks.add_task(_background_save_and_update, uid, room_id, session_id, reply, {"risk_level": risk_level, "safety_response": True, "algorithm_result": algorithm_result}, next_summary)
        _log_professor_demo(user_text, reply, algorithm_result)
        return ChatResponse(
            reply=reply,
            ui_metadata=UIMetadata(sentiment_score=1, suggested_action="Buka menu Konselor", is_risky=True),
            clinical_insight=ClinicalInsight(risk_level=risk_level),
            session_summary=next_summary,
            room_id=room_id,
            session_id=session_id,
            algorithm_trace=algorithm_result,
            debug_metadata=_debug_metadata(algorithm_result, routed_to="safety_safety_triage", elapsed_ms=int((time.perf_counter() - started_at) * 1000)),
            media=media_results
        )

    # 6. Doctor Consultation Shortcut
    doctor_reply = _doctor_direct_reply()
    if doctor_reply and _is_doctor_consultation_request(user_text):
        next_summary = build_fallback_session_summary(context["session_summary"], llm_user_text, doctor_reply, request.mood_signal or "", risk_level)
        background_tasks.add_task(_background_save_and_update, uid, room_id, session_id, doctor_reply, {"risk_level": risk_level, "doctor_shortcut": True, "algorithm_result": algorithm_result}, next_summary)
        _log_professor_demo(user_text, doctor_reply, algorithm_result)
        return ChatResponse(
            reply=doctor_reply,
            ui_metadata=UIMetadata(sentiment_score=current_sentiment, suggested_action="Buka menu Doctor", is_risky=risk_level in {"high", "critical"}),
            clinical_insight=ClinicalInsight(risk_level=risk_level),
            session_summary=next_summary,
            room_id=room_id,
            session_id=session_id,
            algorithm_trace=algorithm_result,
            debug_metadata=_debug_metadata(algorithm_result, routed_to="doctor_consultation_shortcut", elapsed_ms=int((time.perf_counter() - started_at) * 1000)),
            media=media_results
        )

    # 7. LLM Generation
    bot_result = generate_dialog(
        user_message=llm_user_text,
        screening_context=context["latest_screening_summary"],
        session_summary=context["session_summary"],
        profile_context=context["profile_context"],
        memory_context=context["memory_context"],
        recent_daily_context=context["recent_daily_context"],
        risk_level=risk_level,
        mood_signal=request.mood_signal or "",
        user_name=context["name"],
        history_text=history_text,
        keywords=algorithm_result["keywords"],
        relevant_diary=algorithm_result["relevant_diary"],
        style_plan=style_plan,
        emotion_profile=algorithm_result.get("emotion_profile"),
        cognitive_distortions=algorithm_result.get("cognitive_distortions"),
        coping_pathway=algorithm_result.get("coping_pathway"),
        client_time_context=time_context["prompt"],
    )

    reply = bot_result.get("reply") or "Aku dengerin, ya. Bisa ceritain sedikit lagi?"
    next_summary = bot_result.get("session_summary") or build_fallback_session_summary(context["session_summary"], llm_user_text, reply, request.mood_signal or "", risk_level)

    background_tasks.add_task(
        _background_save_and_update,
        uid, room_id, session_id, reply,
        {
            "risk_level": risk_level,
            "sentiment_score": current_sentiment,
            "trend": trend,
            "routing": routing_mode,
            "algorithm_result": algorithm_result,
            "media_analysis": media_results,
        },
        next_summary,
    )

    _log_professor_demo(user_text, reply, algorithm_result)
    
    return ChatResponse(
        reply=reply,
        ui_metadata=UIMetadata(sentiment_score=current_sentiment, suggested_action=bot_result.get("suggested_action"), is_risky=bot_result.get("risk_flag", False)),
        clinical_insight=ClinicalInsight(detected_symptoms=bot_result.get("detected_symptoms", []), dass_category=bot_result.get("dominant_category", "None"), risk_level=risk_level),
        session_summary=next_summary,
        room_id=room_id,
        session_id=session_id,
        algorithm_trace=algorithm_result,
        debug_metadata=_debug_metadata(algorithm_result, routed_to=routing_mode, elapsed_ms=int((time.perf_counter() - started_at) * 1000)),
        media=media_results
    )


@router.post("/finish/", response_model=ChatResponse)
async def finish_chat_endpoint(
    request: ChatFinishRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    uid = current_user["uid"]
    ensure_user_document(uid, current_user)
    existing_session = get_session(uid, request.room_id, request.session_id)
    existing_summary = clean_diary_summary(existing_session.get("summary") or "", "Sesi percakapan selesai.")
    if existing_session.get("status") in {"finished", "finishing"}:
        cleaned_existing_summary = existing_summary
        return ChatResponse(
            reply=cleaned_existing_summary,
            ui_metadata=UIMetadata(sentiment_score=3, suggested_action=None, is_risky=False),
            clinical_insight=ClinicalInsight(),
            session_summary=cleaned_existing_summary,
            room_id=request.room_id,
            session_id=request.session_id,
            gamification=None,
        )

    mark_session_finishing(uid, request.room_id, request.session_id)

    messages = get_session_messages(uid, request.room_id, request.session_id)
    session_raw = format_messages(messages)
    context = get_chat_context(uid, request.room_id, request.session_id)
    final_summary = generate_summary(
        session_raw=session_raw,
        session_summary=context["session_summary"],
        user_name=context["name"],
    )
    finish_session(uid, request.room_id, request.session_id, final_summary)
    create_notification(
        uid=uid,
        title="Diary baru tersimpan",
        body="Sesi chat kamu sudah dirangkum menjadi diary. Kamu bisa membukanya dari kalender atau halaman diary.",
        notification_type="wellbeing",
        priority="low",
        category_label="Diary",
        action_link=f"/diary/{request.room_id}",
        notification_key=f"chat_finished:{request.room_id}:{request.session_id}",
    )

    # Lunar Aura Gamification - Diary Submission
    from services.gamification_service import award_xp
    xp_gained = 20
    is_deep = len(session_raw) > 200
    if is_deep:
        xp_gained += 15
        
    # Get last sentiment score from context or logic
    # In finish_chat, we can retrieve the sentiment score from the last message
    last_msg = messages[-1] if messages else {}
    last_analysis = last_msg.get("analysis_results", {})
    
    # Safety check: sometimes analysis_results might be a string (JSON) or None
    if isinstance(last_analysis, str):
        try:
            last_analysis = json.loads(last_analysis)
        except:
            last_analysis = {}
    elif not isinstance(last_analysis, dict):
        last_analysis = {}

    sentiment_score = last_analysis.get("sentiment_score", 3)
    if sentiment_score == 5 or sentiment_score == 1:
        xp_gained += 10
        
    gamification = award_xp(uid, xp_gained, source="diary", details={"is_deep_reflection": is_deep})

    return ChatResponse(
        reply=final_summary,
        ui_metadata=UIMetadata(sentiment_score=3, suggested_action=None, is_risky=False),
        clinical_insight=ClinicalInsight(),
        session_summary=final_summary,
        room_id=request.room_id,
        session_id=request.session_id,
        gamification=gamification,
    )
