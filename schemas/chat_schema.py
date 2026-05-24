from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    text: Optional[str] = Field("", description="User message")
    room_id: Optional[str] = None
    session_id: Optional[str] = None
    mood_signal: Optional[str] = ""
    client_timezone: Optional[str] = None
    client_utc_offset: Optional[str] = None
    client_local_datetime: Optional[str] = None
    mode: Optional[str] = "chat"
    has_image: bool = False
    media_ids: List[str] = Field(default_factory=list)


class ChatFinishRequest(BaseModel):
    room_id: str
    session_id: str


class UIMetadata(BaseModel):
    sentiment_score: int
    suggested_action: Optional[str] = None
    is_risky: bool


class ClinicalInsight(BaseModel):
    detected_symptoms: List[str] = Field(default_factory=list)
    dass_category: str = "None"
    risk_level: str = "low"


class ChatResponse(BaseModel):
    reply: str
    ui_metadata: UIMetadata
    clinical_insight: ClinicalInsight
    session_summary: str
    room_id: Optional[str] = None
    session_id: Optional[str] = None
    gamification: Optional[Dict[str, Any]] = None
    algorithm_trace: Optional[Dict[str, Any]] = None
    debug_metadata: Optional[Dict[str, Any]] = None
    media: Optional[List[Dict[str, Any]]] = None


class UserContextResponse(BaseModel):
    profile_context: str
    latest_screening_summary: str
    latest_diary_summary: str
    past_diaries: List[str] = Field(default_factory=list)
    has_screening_today: bool
