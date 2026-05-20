from typing import List, Optional

from pydantic import BaseModel, Field


class CalendarSummaryItem(BaseModel):
    date: str
    has_sleep_data: bool = False
    mood: Optional[str] = None
    has_diary: bool = False


class CalendarSummaryResponse(BaseModel):
    items: List[CalendarSummaryItem] = Field(default_factory=list)


class CalendarSleepDetail(BaseModel):
    total_sleep_hours: float = 0
    sleep_quality: str = ""
    bedtime: Optional[str] = None
    wakeup: Optional[str] = None


class CalendarDetailResponse(BaseModel):
    date: str
    mood: Optional[str] = None
    sleep: Optional[CalendarSleepDetail] = None
    diary_snippet: Optional[str] = None
