from fastapi import APIRouter, HTTPException
from schemas.chat_schema import ChatRequest, ChatResponse, UIMetadata, ClinicalInsight
from services.nlp_service import extract_keywords, find_relevant_diary, calculate_risk_level
from services.llm_service import analyze_symptoms_llm, generate_dialog, generate_summary

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Check if mode is summary
        if request.mode == "summary":
            summary_reply = generate_summary(
                session_raw=request.session_raw,
                session_summary=request.session_summary,
                user_name=request.user_name
            )
            return ChatResponse(
                reply=summary_reply,
                ui_metadata=UIMetadata(sentiment_score=3, suggested_action=None, is_risky=False),
                clinical_insight=ClinicalInsight(),
                session_summary=request.session_summary
            )

        # 2. Regular Chat Logic
        # Risk assessment
        risk_level = calculate_risk_level(
            text=request.text,
            screening_context=request.screening_context,
            session_summary=request.session_summary,
            client_risk=request.risk_level
        )

        if risk_level == "high":
            reply = (
                "Aku sangat peduli sama kamu, dan sepertinya kondisimu sedang sangat berat. "
                "Kamu nggak sendirian ya. Sereluna punya fitur Konselor di mana kamu bisa ngobrol langsung dengan ahlinya. "
                "Coba cek menu Konselor ya, ada yang siap dengerin kamu. Aku juga bakal tetap nemenin kamu di sini."
            )
            return ChatResponse(
                reply=reply,
                ui_metadata=UIMetadata(sentiment_score=1, suggested_action="Buka menu Konselor", is_risky=True),
                clinical_insight=ClinicalInsight(risk_level="high"),
                session_summary=request.session_summary
            )

        # NLP Services: Keywords and Relevant Diary
        keywords = extract_keywords(request.text)
        relevant_diary = find_relevant_diary(request.text, request.past_diaries or [])

        # LLM Services: Symptom analysis
        analysis_result = analyze_symptoms_llm(request.text)
        
        # LLM Services: Dialog generation
        # We use session_raw or some other way to represent history if needed. 
        # For now, let's pass what we have.
        bot_result = generate_dialog(
            user_message=request.text,
            analysis_data=analysis_result,
            screening_context=request.screening_context,
            session_summary=request.session_summary,
            profile_context=request.profile_context,
            risk_level=risk_level,
            mood_signal=request.mood_signal,
            user_name=request.user_name,
            history_text=request.session_raw, # Using session_raw as history for now
            keywords=keywords,
            relevant_diary=relevant_diary
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
            session_summary=request.session_summary # In a real app, you might update this
        )

    except Exception as e:
        # Mimic GAS error handling
        return ChatResponse(
            reply="Duh, maaf ya teman, otakku lagi agak lemot karena koneksi. Bisa tolong ceritain ulang?",
            ui_metadata=UIMetadata(sentiment_score=3, suggested_action=None, is_risky=False),
            clinical_insight=ClinicalInsight(),
            session_summary=request.session_summary
        )
