from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScreeningRequest(BaseModel):
    answers: List[int] = Field(..., min_items=21, max_items=21)
    note: Optional[str] = ""


class ScreeningResponse(BaseModel):
    date: str
    scores: Dict[str, int]
    severity: Dict[str, str]
    summary: str
    algorithm: Dict[str, Any] = Field(default_factory=dict)
    has_screening_today: bool = True
    next_recommended_date: Optional[str] = None
    next_recommended_in_days: Optional[int] = None
    recommended_interval_days: int = 7
    updated_at: Optional[str] = None
    updated_statistics_version: Optional[str] = None
    disclaimer: str = "DASS-21 adalah alat screening, bukan diagnosis medis."


class ScreeningQuestion(BaseModel):
    id: int
    category: str
    text: str
    answer_min: int = 0
    answer_max: int = 3


class ScreeningAnswerOption(BaseModel):
    value: int
    label: str


class ScreeningQuestionnaireResponse(BaseModel):
    instrument: str
    version: str
    source_file: str
    recommended_interval_days: int
    disclaimer: str
    instructions: str
    answer_options: List[ScreeningAnswerOption]
    questions: List[ScreeningQuestion]


class ScreeningStatusResponse(BaseModel):
    instrument: str
    recommended_interval_days: int
    is_due: bool
    latest: Optional[Dict[str, Any]] = None
    next_recommended_date: Optional[str] = None
    next_recommended_in_days: int = 0
    server_time: str
    updated_at: str
    disclaimer: str
