from typing import List, Optional

from pydantic import BaseModel, Field


class DiarySessionItem(BaseModel):
    id: str
    model: str = ""
    title: str = ""
    content: str = ""
    summary: str = ""
    preview: str = ""
    status: str = ""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    updated_at: Optional[str] = None


class DiaryItem(BaseModel):
    id: str
    date: str = ""
    title: str = ""
    content: str = ""
    chat_summary: str = ""
    preview: str = ""
    session_count: int = 0
    sessions: List[DiarySessionItem] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DiaryListResponse(BaseModel):
    items: List[DiaryItem] = Field(default_factory=list)


class DiaryEntryItem(BaseModel):
    id: str
    diary_id: str
    session_id: str
    date: str = ""
    title: str = ""
    content: str = ""
    summary: str = ""
    preview: str = ""
    status: str = ""
    model: str = ""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    updated_at: Optional[str] = None


class DiaryEntryListResponse(BaseModel):
    items: List[DiaryEntryItem] = Field(default_factory=list)


class DiaryDetailResponse(BaseModel):
    id: str
    date: str = ""
    title: str = ""
    content: str = ""
    chat_summary: str = ""
    preview: str = ""
    session_count: int = 0
    sessions: List[DiarySessionItem] = Field(default_factory=list)


class DiaryMessageItem(BaseModel):
    id: str
    sender_role: str
    text: str = ""
    timestamp: Optional[str] = None


class DiaryMessagesResponse(BaseModel):
    items: List[DiaryMessageItem] = Field(default_factory=list)
