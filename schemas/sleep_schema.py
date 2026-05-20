from typing import List, Optional

from pydantic import BaseModel, Field


class SleepDailyRequest(BaseModel):
    date: str
    bedtime: str
    wakeup: str
    sleep_quality: str
    total_sleep_hours: float


class SleepDailyResponse(BaseModel):
    success: bool = True


class SleepDailyItem(BaseModel):
    date: str
    sleep_quality: str = ""
    total_sleep_hours: float = 0
    bedtime: Optional[str] = None
    wakeup: Optional[str] = None
    updated_at: Optional[str] = None


class SleepDailyListResponse(BaseModel):
    items: List[SleepDailyItem] = Field(default_factory=list)
