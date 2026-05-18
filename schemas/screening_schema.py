from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ScreeningRequest(BaseModel):
    answers: List[int] = Field(..., min_items=21, max_items=21)
    note: Optional[str] = ""


class ScreeningResponse(BaseModel):
    date: str
    scores: Dict[str, int]
    severity: Dict[str, str]
    summary: str
    has_screening_today: bool = True
