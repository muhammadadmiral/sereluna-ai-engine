import asyncio
from services.llm_service import generate_dialog
import json

def run_tests():
    test_cases = [
        {
            "name": "1. Casual Greeting",
            "msg": "halo mek selamat malem",
            "risk": "low",
            "intent": "check_in",
            "response_mode": "low_signal_greeting"
        },
        {
            "name": "2. Off-Domain",
            "msg": "bikinin kodingan python dong buat scraping",
            "risk": "low",
            "intent": "off_domain_redirect",
            "response_mode": "boundary_redirect"
        },
        {
            "name": "3. Moderate Distress (Data Gathering)",
            "msg": "gua lagi sedih banget hari ini, rasanya campur aduk",
            "risk": "medium",
            "intent": "emotional_support",
            "response_mode": "assessment_response"
        },
        {
            "name": "4. High Risk (Crisis)",
            "msg": "gua udah ga kuat lagi, rasanya pengen mati aja",
            "risk": "high",
            "intent": "safety_support",
            "response_mode": "crisis_response"
        }
    ]

    for tc in test_cases:
        print(f"\n--- TESTING: {tc['name']} ---")
        style_plan = {
            "intent": tc["intent"],
            "response_mode": tc["response_mode"],
            "user_register": "gue-lu",
            "user_style_profile": {"register": "gue-lu"}
        }
        res = generate_dialog(
            user_message=tc["msg"],
            screening_context="",
            session_summary="",
            profile_context="",
            memory_context="",
            recent_daily_context="",
            risk_level=tc["risk"],
            mood_signal="sad",
            user_name="TestingUser",
            history_text="",
            keywords=[],
            style_plan=style_plan
        )
        print("BOT:", res['reply'])

if __name__ == "__main__":
    run_tests()
