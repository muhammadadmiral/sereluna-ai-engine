import json
import logging
import time
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from schemas.chat_schema import ChatFinishRequest, ChatRequest, ChatResponse, ClinicalInsight, UIMetadata
from services.context_service import (
    ensure_user_document,
    finish_session,
    format_messages,
    get_chat_context,
    get_or_create_session,
    get_or_create_today_diary,
    get_session_messages,
    save_message,
    update_chat_summaries,
)
from services.firebase_service import get_current_user
from services.llm_service import build_fallback_session_summary, generate_dialog, generate_summary
from services.media_service import analyze_images_for_chat
from services.nlp_service import build_context_algorithm_result, build_response_style_plan
from services.notification_service import create_notification

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
logger = logging.getLogger("sereluna.chat")
logger.setLevel(logging.INFO)

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
        acc_std = eval_data.get("accuracy_std", "N/A")
        f1_macro = eval_data.get("macro_f1", "N/A")
        train_size = eval_data.get("train_size", "N/A")
        dataset_path = eval_data.get("dataset_path", "N/A")

        preprocessing_filter = algorithm_result.get("preprocessing_filter", {})
        filter_method = (preprocessing_filter.get("algorithm", {})).get("method", "N/A")
        normalized = preprocessing_filter.get("normalized_text", "N/A")

        supervised = algorithm_result.get("supervised_emotion_classifier") or {}
        predicted_emotion = supervised.get("predicted_emotion", "N/A")
        confidence = supervised.get("confidence", 0.0)
        risk_level = algorithm_result.get("risk_level", "unknown")

        log_msg = "\n" + "="*70 + "\n"
        log_msg += f"🤖 [SERELUNA AI - DEMO DOSEN - BACKEND PROCESS] 🤖\n"
        log_msg += "="*70 + "\n"
        log_msg += f"1. DATA MINING & PREPROCESSING\n"
        log_msg += f"   - Input Teks     : {user_text}\n"
        log_msg += f"   - Metode Filter  : {filter_method}\n"
        log_msg += f"   - Hasil Normalisasi: {normalized}\n"
        log_msg += "-"*70 + "\n"
        log_msg += f"2. MACHINE LEARNING (EMOTION CLASSIFICATION)\n"
        log_msg += f"   - Model Dataset  : {dataset_path} ({train_size} baris training)\n"
        log_msg += f"   - K-Fold Accuracy : {accuracy} (±{acc_std})\n"
        log_msg += f"   - Macro F1 Score : {f1_macro}\n"
        log_msg += f"   - Prediksi Emosi : {predicted_emotion.upper()} (Confidence: {confidence})\n"
        log_msg += "-"*70 + "\n"
        log_msg += f"3. LOGIC BERAT & RISK ASSESSMENT\n"
        log_msg += f"   - Klasifikasi Risiko: {risk_level.upper()}\n"
        log_msg += f"   - Coping Pathway : {(algorithm_result.get('coping_pathway', {})).get('pathway', 'N/A')}\n"
        log_msg += "-"*70 + "\n"
        log_msg += f"4. LLM GENERATION RESULT\n"
        log_msg += f"   - Output Bot     : {reply}\n"
        log_msg += "="*70 + "\n"
        
        logger.info(log_msg)
        # Fallback print for direct console visibility during demo
        print(log_msg)
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

    mode = (request.mode or "chat").lower()
    if mode != "chat":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use POST /api/v1/chat/finish/ to finish a chat session",
        )

    uid = current_user["uid"]
    _log_chat_pipeline(
        trace_id,
        "request_received",
        {
            "uid": uid,
            "room_id": request.room_id,
            "session_id": request.session_id,
            "message_length": len(user_text),
            "mood_signal": request.mood_signal or "",
            "media_count": len(media_ids),
        },
    )
    ensure_user_document(uid, current_user)
    room_id, _diary = get_or_create_today_diary(uid, request.room_id)
    session_id, _session = get_or_create_session(uid, room_id, request.session_id)

    media_results = analyze_images_for_chat(uid, media_ids) if media_ids else []
    media_context = _format_media_context(media_results)
    llm_user_text = user_text
    if media_context:
        llm_user_text = (
            f"{user_text}\n\n"
            "[Konteks gambar dari backend vision model]\n"
            f"{media_context}"
        )

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
        },
    )

    context = get_chat_context(uid, room_id, session_id)
    screening_context = context["latest_screening_summary"]
    session_summary = context["session_summary"]
    profile_context = context["profile_context"]
    memory_context = context["memory_context"]
    recent_daily_context = context["recent_daily_context"]
    history_text = context["session_history"]
    past_diaries = context["past_diaries"]

    algorithm_result = build_context_algorithm_result(
        text=llm_user_text,
        mood_signal=request.mood_signal or "",
        screening_context=screening_context,
        session_summary=session_summary,
        past_diaries=past_diaries,
    )
    algorithm_result["recent_daily_context"] = recent_daily_context
    risk_level = algorithm_result["risk_level"]
    _log_chat_pipeline(
        trace_id,
        "backend_algorithm_completed",
        {
            "room_id": room_id,
            "session_id": session_id,
            "risk_level": risk_level,
            "sentiment_score": algorithm_result.get("sentiment_score"),
            "primary_emotion": (algorithm_result.get("emotion_profile") or {}).get("primary_emotion"),
            "supervised_emotion": (algorithm_result.get("supervised_emotion_classifier") or {}).get("predicted_emotion"),
            "supervised_confidence": (algorithm_result.get("supervised_emotion_classifier") or {}).get("confidence"),
            "model_accuracy": (algorithm_result.get("supervised_model_evaluation") or {}).get("accuracy"),
        },
    )
    style_plan = build_response_style_plan(
        text=llm_user_text,
        mood_signal=request.mood_signal or "",
        risk_level=risk_level,
        sentiment_score=algorithm_result["sentiment_score"],
        session_summary=session_summary,
        history_text=history_text,
    )
    algorithm_result["style_plan"] = style_plan
    algorithm_result.setdefault("algorithms", {}).setdefault("supporting", []).append("sereluna_response_planner")
    memory_scope = style_plan.get("memory_scope")
    effective_screening_context = screening_context
    effective_memory_context = memory_context
    effective_recent_daily_context = recent_daily_context
    if memory_scope == "current_room_only":
        effective_screening_context = ""
        effective_memory_context = history_text
        effective_recent_daily_context = ""
    risk_trace = algorithm_result.get("risk", {})
    route_to_safety = risk_trace.get("reason") in {
        "current_crisis_signal",
        "preprocessing_crisis_filter",
    }
    sentiment_score = algorithm_result["sentiment_score"]

    if route_to_safety:
        _log_chat_pipeline(
            trace_id,
            "safety_route_selected",
            {"risk_level": risk_level, "risk_reason": risk_trace.get("reason")},
        )
        reply = _safety_reply()
        next_summary = build_fallback_session_summary(
            previous_summary=session_summary,
            user_message=llm_user_text,
            assistant_reply=reply,
            mood_signal=request.mood_signal or "",
            risk_level=risk_level,
        )
        
        background_tasks.add_task(
            _background_save_and_update,
            uid,
            room_id,
            session_id,
            reply,
            {
                "risk_level": risk_level,
                "safety_response": True,
                "algorithm_result": algorithm_result,
                "risk_reason": risk_trace.get("reason"),
                "media_ids": media_ids,
                "media_analysis": media_results,
            },
            next_summary,
        )
        
        return ChatResponse(
            reply=reply,
            ui_metadata=UIMetadata(
                sentiment_score=min(sentiment_score, 2),
                suggested_action="Buka menu Konselor",
                is_risky=True,
            ),
            clinical_insight=ClinicalInsight(risk_level=risk_level),
            session_summary=next_summary,
            room_id=room_id,
            session_id=session_id,
            algorithm_trace=algorithm_result,
            debug_metadata=_debug_metadata(
                algorithm_result,
                routed_to="safety_reply_without_llm",
                elapsed_ms=int((time.perf_counter() - started_at) * 1000),
            ),
            media=media_results,
        )

    keywords = algorithm_result["keywords"]
    relevant_diary = algorithm_result["relevant_diary"]

    _log_chat_pipeline(
        trace_id,
        "llm_call_started",
        {
            "room_id": room_id,
            "session_id": session_id,
            "risk_level": risk_level,
            "style_target_paragraphs": style_plan.get("desired_paragraphs"),
            "memory_scope": style_plan.get("memory_scope"),
        },
    )
    bot_result = generate_dialog(
        user_message=llm_user_text,
        screening_context=effective_screening_context,
        session_summary=session_summary,
        profile_context=profile_context,
        memory_context=effective_memory_context,
        recent_daily_context=effective_recent_daily_context,
        risk_level=risk_level,
        mood_signal=request.mood_signal or "",
        user_name=context["name"],
        history_text=history_text,
        keywords=keywords,
        relevant_diary=relevant_diary,
        style_plan=style_plan,
        emotion_profile=algorithm_result.get("emotion_profile"),
        cognitive_distortions=algorithm_result.get("cognitive_distortions"),
        coping_pathway=algorithm_result.get("coping_pathway"),
    )
    _log_chat_pipeline(
        trace_id,
        "llm_call_completed",
        {
            "reply_length": len(bot_result.get("reply") or ""),
            "llm_sentiment_score": bot_result.get("sentiment_score"),
            "suggested_action": bot_result.get("suggested_action"),
        },
    )

    reply = bot_result.get("reply") or "Aku dengerin, ya. Bisa ceritain sedikit lagi?"
    next_summary = bot_result.get("session_summary") or build_fallback_session_summary(
        previous_summary=session_summary,
            user_message=llm_user_text,
        assistant_reply=reply,
        mood_signal=request.mood_signal or "",
        risk_level=risk_level,
    )

    background_tasks.add_task(
        _background_save_and_update,
        uid,
        room_id,
        session_id,
        reply,
        {
            "risk_level": risk_level,
            "sentiment_score": sentiment_score,
            "llm_sentiment_score": bot_result.get("sentiment_score"),
            "suggested_action": bot_result.get("suggested_action"),
            "algorithm_result": algorithm_result,
            "media_ids": media_ids,
            "media_analysis": media_results,
        },
        next_summary,
    )

    _log_professor_demo(user_text, reply, algorithm_result)
    return ChatResponse(
        reply=reply,
        ui_metadata=UIMetadata(
            sentiment_score=sentiment_score,
            suggested_action=bot_result.get("suggested_action"),
            is_risky=bot_result.get("risk_flag", False),
        ),
        clinical_insight=ClinicalInsight(
            detected_symptoms=bot_result.get("detected_symptoms", []),
            dass_category=bot_result.get("dominant_category", "None"),
            risk_level=risk_level,
        ),
        session_summary=next_summary,
        room_id=room_id,
        session_id=session_id,
        algorithm_trace=algorithm_result,
        debug_metadata=_debug_metadata(
            algorithm_result,
            routed_to="llm_after_backend_processing",
            elapsed_ms=int((time.perf_counter() - started_at) * 1000),
        ),
        media=media_results,
    )


@router.post("/finish/", response_model=ChatResponse)
async def finish_chat_endpoint(
    request: ChatFinishRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    uid = current_user["uid"]
    ensure_user_document(uid, current_user)

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
