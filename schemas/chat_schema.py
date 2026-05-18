from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    text: Optional[str] = Field("", description="User message")
    room_id: Optional[str] = "default-room"
    screening_context: Optional[str] = ""
    session_summary: Optional[str] = ""
    risk_level: Optional[str] = ""
    mood_signal: Optional[str] = ""
    groq_api_key: Optional[str] = None
    mode: Optional[str] = "chat"
    session_raw: Optional[str] = ""
    user_name: Optional[str] = "Teman"
    profile_context: Optional[str] = ""
    past_diaries: Optional[List[str]] = Field(default_factory=list)

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
