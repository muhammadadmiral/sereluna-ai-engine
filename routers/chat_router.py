from fastapi import APIRouter
from schemas.chat_schema import ChatRequest, ChatResponse, UIMetadata, ClinicalInsight
from services.nlp_service import extract_keywords, find_relevant_diary, calculate_risk_level
from services.llm_service import (
    analyze_symptoms_llm,
    build_fallback_session_summary,
    generate_dialog,
    generate_summary,
)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        mode = (request.mode or "chat").lower()
        user_text = request.text or ""
        screening_context = request.screening_context or ""
        session_summary = request.session_summary or ""
        session_raw = request.session_raw or ""
        mood_signal = request.mood_signal or ""
        risk_signal = request.risk_level or ""
        user_name = request.user_name or "Teman"
        profile_context = request.profile_context or ""

        # 1. Check if mode is summary
        if mode == "summary":
            summary_reply = generate_summary(
                session_raw=session_raw,
                session_summary=session_summary,
                user_name=user_name,
                groq_api_key=request.groq_api_key,
            )
            return ChatResponse(
                reply=summary_reply,
                ui_metadata=UIMetadata(sentiment_score=3, suggested_action=None, is_risky=False),
                clinical_insight=ClinicalInsight(),
                session_summary=summary_reply
            )

        # 2. Regular Chat Logic
        # Risk assessment
        risk_level = calculate_risk_level(
            text=user_text,
            screening_context=screening_context,
            session_summary=session_summary,
            client_risk=risk_signal
        )

        if risk_level == "high":
            reply = (
                "Aku sangat peduli sama kamu, dan sepertinya kondisimu sedang sangat berat. "
                "Kamu nggak sendirian ya. Sereluna punya fitur Konselor di mana kamu bisa ngobrol langsung dengan ahlinya. "
                "Coba cek menu Konselor ya, ada yang siap dengerin kamu. Aku juga bakal tetap nemenin kamu di sini."
            )
            next_summary = build_fallback_session_summary(
                previous_summary=session_summary,
                user_message=user_text,
                assistant_reply=reply,
                mood_signal=mood_signal,
                risk_level=risk_level,
            )
            return ChatResponse(
                reply=reply,
                ui_metadata=UIMetadata(sentiment_score=1, suggested_action="Buka menu Konselor", is_risky=True),
                clinical_insight=ClinicalInsight(risk_level="high"),
                session_summary=next_summary
            )

        # NLP Services: Keywords and Relevant Diary
        keywords = extract_keywords(user_text)
        relevant_diary = find_relevant_diary(user_text, request.past_diaries or [])

        # LLM Services: Symptom analysis
        analysis_result = analyze_symptoms_llm(user_text, groq_api_key=request.groq_api_key)
        
        # LLM Services: Dialog generation
        # We use session_raw or some other way to represent history if needed. 
        # For now, let's pass what we have.
        bot_result = generate_dialog(
            user_message=user_text,
            analysis_data=analysis_result,
            screening_context=screening_context,
            session_summary=session_summary,
            profile_context=profile_context,
            risk_level=risk_level,
            mood_signal=mood_signal,
            user_name=user_name,
            history_text=session_raw, # Using session_raw as history for now
            keywords=keywords,
            relevant_diary=relevant_diary,
            groq_api_key=request.groq_api_key,
        )

        return ChatResponse(
            reply=bot_result.get("reply", ""),
            ui_metadata=UIMetadata(
                sentiment_score=bot_result.get("sentiment_score", 3),
                suggested_action=bot_result.get("suggested_action"),
                is_risky=bot_result.get("risk_flag", False)
            ),
            clinical_insight=ClinicalInsight(
                detected_symptoms=analysis_result.get("detected_symptoms", []),
                dass_category=analysis_result.get("dominant_category", "None"),
                risk_level=risk_level
            ),
            session_summary=bot_result.get("session_summary") or session_summary
        )

    except Exception as e:
        # Mimic GAS error handling
        fallback_reply = "Duh, maaf ya teman, otakku lagi agak lemot karena koneksi. Bisa tolong ceritain ulang?"
        return ChatResponse(
            reply=fallback_reply,
            ui_metadata=UIMetadata(sentiment_score=3, suggested_action=None, is_risky=False),
            clinical_insight=ClinicalInsight(),
            session_summary=build_fallback_session_summary(
                previous_summary=request.session_summary or "",
                user_message=request.text,
                assistant_reply=fallback_reply,
                mood_signal=request.mood_signal or "",
                risk_level=request.risk_level or "low",
            )
        )
