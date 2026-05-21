import asyncio
from services.llm_service import generate_dialog
import json

res = generate_dialog(
    user_message="masa urai bareng terus",
    screening_context="",
    session_summary="",
    profile_context="",
    memory_context="",
    recent_daily_context="",
    risk_level="low",
    mood_signal="neutral",
    user_name="Muhammad Admiral",
    history_text="User: benar kah ini sereluna ai? \nSereluna: Aku dengerin, ya.",
    keywords=["urai", "bareng"],
    style_plan={
        "intent":"reflective_companion",
        "emotional_intensity":"low",
        "relationship_stage":"new_room",
        "assistant_turns":0,
        "user_register":"aku-kamu santai",
        "desired_paragraphs":2,
        "target_words":{"minimum":70,"maximum":170}
    }
)
print(json.dumps(res, indent=2))
