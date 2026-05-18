from typing import List, Optional

from pydantic import BaseModel, Field


class NotificationItem(BaseModel):
    id: str
    title: str = ""
    body: str = ""
    type: str = ""
    is_read: bool = False
    created_at: Optional[str] = None


class NotificationListResponse(BaseModel):
    items: List[NotificationItem] = Field(default_factory=list)


class SuccessResponse(BaseModel):
    success: bool = True
