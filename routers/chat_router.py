from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

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
from services.llm_service import analyze_symptoms_llm, build_fallback_session_summary, generate_dialog, generate_summary
from services.nlp_service import build_context_algorithm_result, build_response_style_plan

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _safety_reply() -> str:
    return (
        "Aku sangat peduli sama kamu, dan sepertinya kondisimu sedang sangat berat. "
        "Kamu nggak sendirian ya. Sereluna punya fitur Konselor di mana kamu bisa ngobrol langsung dengan ahlinya. "
        "Coba cek menu Konselor ya, ada yang siap dengerin kamu. Aku juga bakal tetap nemenin kamu di sini."
    )


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_text = (request.text or "").strip()
    if not user_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="text is required")

    mode = (request.mode or "chat").lower()
    if mode != "chat":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use POST /api/v1/chat/finish/ to finish a chat session",
        )

    uid = current_user["uid"]
    ensure_user_document(uid, current_user)
    room_id, _diary = get_or_create_today_diary(uid, request.room_id)
    session_id, _session = get_or_create_session(uid, room_id, request.session_id)

    save_message(
        uid=uid,
        diary_id=room_id,
        session_id=session_id,
        role="user",
        text=user_text,
        metadata={"mood_signal": request.mood_signal or ""},
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
        text=user_text,
        mood_signal=request.mood_signal or "",
        screening_context=screening_context,
        session_summary=session_summary,
        past_diaries=past_diaries,
    )
    algorithm_result["recent_daily_context"] = recent_daily_context
    risk_level = algorithm_result["risk_level"]
    style_plan = build_response_style_plan(
        text=user_text,
        mood_signal=request.mood_signal or "",
        risk_level=risk_level,
        sentiment_score=algorithm_result["sentiment_score"],
        session_summary=session_summary,
        history_text=history_text,
    )
    algorithm_result["style_plan"] = style_plan
    algorithm_result.setdefault("algorithms", {}).setdefault("supporting", []).append("sereluna_response_planner")
    risk_trace = algorithm_result.get("risk", {})
    route_to_safety = risk_trace.get("reason") in {
        "current_crisis_signal",
        "screening_crisis_signal",
    }
    sentiment_score = algorithm_result["sentiment_score"]

    if route_to_safety:
        reply = _safety_reply()
        next_summary = build_fallback_session_summary(
            previous_summary=session_summary,
            user_message=user_text,
            assistant_reply=reply,
            mood_signal=request.mood_signal or "",
            risk_level=risk_level,
        )
        save_message(
            uid=uid,
            diary_id=room_id,
            session_id=session_id,
            role="assistant",
            text=reply,
            metadata={
                "risk_level": risk_level,
                "safety_response": True,
                "algorithm_result": algorithm_result,
                "risk_reason": risk_trace.get("reason"),
            },
        )
        update_chat_summaries(uid, room_id, session_id, next_summary)
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
        )

    keywords = algorithm_result["keywords"]
    relevant_diary = algorithm_result["relevant_diary"]
    analysis_result = analyze_symptoms_llm(user_text)
    bot_result = generate_dialog(
        user_message=user_text,
        analysis_data=analysis_result,
        screening_context=screening_context,
        session_summary=session_summary,
        profile_context=profile_context,
        memory_context=memory_context,
        recent_daily_context=recent_daily_context,
        risk_level=risk_level,
        mood_signal=request.mood_signal or "",
        user_name=context["name"],
        history_text=history_text,
        keywords=keywords,
        relevant_diary=relevant_diary,
        style_plan=style_plan,
    )

    reply = bot_result.get("reply") or "Aku dengerin, ya. Bisa ceritain sedikit lagi?"
    next_summary = bot_result.get("session_summary") or build_fallback_session_summary(
        previous_summary=session_summary,
        user_message=user_text,
        assistant_reply=reply,
        mood_signal=request.mood_signal or "",
        risk_level=risk_level,
    )

    save_message(
        uid=uid,
        diary_id=room_id,
        session_id=session_id,
        role="assistant",
        text=reply,
        metadata={
            "risk_level": risk_level,
            "sentiment_score": sentiment_score,
            "llm_sentiment_score": bot_result.get("sentiment_score"),
            "suggested_action": bot_result.get("suggested_action"),
            "algorithm_result": algorithm_result,
        },
    )
    update_chat_summaries(uid, room_id, session_id, next_summary)

    return ChatResponse(
        reply=reply,
        ui_metadata=UIMetadata(
            sentiment_score=sentiment_score,
            suggested_action=bot_result.get("suggested_action"),
            is_risky=bot_result.get("risk_flag", False),
        ),
        clinical_insight=ClinicalInsight(
            detected_symptoms=analysis_result.get("detected_symptoms", []),
            dass_category=analysis_result.get("dominant_category", "None"),
            risk_level=risk_level,
        ),
        session_summary=next_summary,
        room_id=room_id,
        session_id=session_id,
        algorithm_trace=algorithm_result,
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

    return ChatResponse(
        reply=final_summary,
        ui_metadata=UIMetadata(sentiment_score=3, suggested_action=None, is_risky=False),
        clinical_insight=ClinicalInsight(),
        session_summary=final_summary,
        room_id=request.room_id,
        session_id=request.session_id,
    )
