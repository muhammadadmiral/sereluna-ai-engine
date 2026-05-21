from pydantic import BaseModel
from typing import Optional


class MoodDailyRequest(BaseModel):
    date: str
    mood: str


class MoodDailyResponse(BaseModel):
    success: bool = True
    updated_at: Optional[str] = None
    updated_statistics_version: Optional[str] = None
