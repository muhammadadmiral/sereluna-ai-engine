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


class WellbeingDailyItem(BaseModel):
    date: str
    mood: Optional[str] = None
    wellbeing_score: Optional[int] = None
    wellbeing_level: str = "no_data"
    risk_level: str = "low"


class WellbeingStatisticsResponse(BaseModel):
    range: str
    period_days: int
    overall_mood: str
    average_wellbeing_score: Optional[float] = None
    mood_distribution: Dict[str, int]
    mood_distribution_detail: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    dominant_mood: Optional[str] = None
    screening_context: Optional[Dict[str, Any]] = None
    insights: List[str]
    daily_items: List[WellbeingDailyItem]
    model_version: str
    updated_at: Optional[str] = None
    updated_statistics_version: Optional[str] = None
    disclaimer: str = "Insight ini bukan diagnosis medis."
