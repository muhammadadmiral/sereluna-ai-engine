import re
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from schemas.mood_schema import MoodDailyRequest, MoodDailyResponse
from services.daily_dashboard_service import MOOD_VALUES, save_daily_mood
from services.firebase_service import get_current_user
from services.notification_service import create_notification

router = APIRouter(prefix="/mood", tags=["mood"])
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@router.post("/", response_model=MoodDailyResponse)
async def save_mood_daily(
    request: MoodDailyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    date_value = request.date.strip()
    mood = request.mood.strip().lower()
    if not DATE_PATTERN.match(date_value):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="date must use yyyy-MM-dd format")
    if mood not in MOOD_VALUES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mood must be one of: happy, neutral, sad, anxious, angry",
        )

    save_daily_mood(uid=current_user["uid"], date=date_value, mood=mood)
    create_notification(
        uid=current_user["uid"],
        title="Mood harian tersimpan",
        body="Mood kamu sudah masuk ke kalender wellbeing hari ini.",
        notification_type="wellbeing",
        priority="low",
        category_label="Kalender",
        action_link=f"/calendar?date={date_value}",
        notification_key=f"mood_saved:{date_value}",
    )
    updated_at = _utc_now_iso()
    return MoodDailyResponse(
        success=True,
        updated_at=updated_at,
        updated_statistics_version=updated_at,
    )
