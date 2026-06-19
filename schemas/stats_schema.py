from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MoodDistributionItem(BaseModel):
    mood: str
    count: int


class MoodDistributionResponse(BaseModel):
    period_days: int
    data: List[MoodDistributionItem]
    dominant_mood: Optional[str] = None
    insight: str


class SleepTrendItem(BaseModel):
    date: str
    hours: float


class SleepTrendResponse(BaseModel):
    average_hours: float
    items: List[SleepTrendItem]
    insight: str
