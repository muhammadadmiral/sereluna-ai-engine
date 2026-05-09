from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    text: str = Field(..., description="User message")
    room_id: Optional[str] = "default-room"
    screening_context: Optional[str] = ""
    session_summary: Optional[str] = ""
    risk_level: Optional[str] = ""
    mood_signal: Optional[str] = ""
    mode: Optional[str] = "chat"
    session_raw: Optional[str] = ""
    user_name: Optional[str] = "Teman"
    profile_context: Optional[str] = ""
    past_diaries: Optional[List[str]] = []

class UIMetadata(BaseModel):
    sentiment_score: int
    suggested_action: Optional[str] = None
    is_risky: bool

class ClinicalInsight(BaseModel):
    detected_symptoms: List[str] = []
    dass_category: str = "None"
    risk_level: str = "low"

class ChatResponse(BaseModel):
    reply: str
    ui_metadata: UIMetadata
    clinical_insight: ClinicalInsight
    session_summary: str
