from typing import Optional

from pydantic import BaseModel


class ProfileResponse(BaseModel):
    uid: str
    name: str = ""
    email: str = ""
    photo_url: str = ""
    provider: str = ""
    latest_screening_summary: str = ""
    latest_diary_summary: str = ""
    personal_context: str = ""
    has_screening_today: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    photo_url: Optional[str] = None
