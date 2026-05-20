from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CalendarSummaryItem(BaseModel):
    date: str
    has_sleep_data: bool = False
    mood: Optional[str] = None
    has_diary: bool = False
    wellbeing_score: Optional[int] = None
    wellbeing_level: str = "no_data"
    indicator: str = "empty"


class CalendarSummaryResponse(BaseModel):
    items: List[CalendarSummaryItem] = Field(default_factory=list)


class CalendarSleepDetail(BaseModel):
    total_sleep_hours: float = 0
    sleep_quality: str = ""
    bedtime: Optional[str] = None
    wakeup: Optional[str] = None


class CalendarWellbeingComponent(BaseModel):
    name: str
    score: int
    weight: float
    reason: str = ""


class CalendarWellbeingInsight(BaseModel):
    score: Optional[int] = None
    level: str = "no_data"
    signals: List[str] = Field(default_factory=list)
    recommendation: Optional[str] = None
    components: List[CalendarWellbeingComponent] = Field(default_factory=list)
    algorithm: Dict[str, Any] = Field(default_factory=dict)


class CalendarDetailResponse(BaseModel):
    date: str
    mood: Optional[str] = None
    sleep: Optional[CalendarSleepDetail] = None
    diary_snippet: Optional[str] = None
    wellbeing: CalendarWellbeingInsight = Field(default_factory=CalendarWellbeingInsight)
