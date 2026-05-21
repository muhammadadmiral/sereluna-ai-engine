from typing import List, Optional

from pydantic import BaseModel, Field


class NotificationItem(BaseModel):
    id: str
    title: str = ""
    body: str = ""
    type: str = ""
    priority: str = "low"
    category_label: str = "Sistem"
    is_read: bool = False
    created_at: Optional[str] = None
    read_at: Optional[str] = None
    action_link: Optional[str] = None


class NotificationListResponse(BaseModel):
    items: List[NotificationItem] = Field(default_factory=list)
    unread_count: int = 0
    updated_at: Optional[str] = None


class SuccessResponse(BaseModel):
    success: bool = True
    unread_count: Optional[int] = None
    updated_at: Optional[str] = None


class UnreadCountResponse(BaseModel):
    unread_count: int
    updated_at: str
