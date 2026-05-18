from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, status

from schemas.notification_schema import NotificationListResponse, SuccessResponse
from services.firebase_service import get_current_user
from services.notification_service import list_notifications, mark_notification_read

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/", response_model=NotificationListResponse)
async def read_notifications(
    limit: int = Query(30, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return NotificationListResponse(items=list_notifications(current_user["uid"], limit))


@router.patch("/{notification_id}/read/", response_model=SuccessResponse)
async def mark_read(
    notification_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    updated = mark_notification_read(current_user["uid"], notification_id)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    return SuccessResponse(success=True)
