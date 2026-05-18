from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, status

from schemas.diary_schema import DiaryDetailResponse, DiaryListResponse, DiaryMessagesResponse
from services.diary_service import get_diary_detail, list_diaries, list_session_messages
from services.firebase_service import get_current_user

router = APIRouter(prefix="/diaries", tags=["diaries"])


@router.get("/", response_model=DiaryListResponse)
async def read_diaries(
    limit: int = Query(30, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return DiaryListResponse(items=list_diaries(current_user["uid"], limit))


@router.get("/{diary_id}/", response_model=DiaryDetailResponse)
async def read_diary_detail(
    diary_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    detail = get_diary_detail(current_user["uid"], diary_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diary not found")
    return DiaryDetailResponse(**detail)


@router.get("/{diary_id}/sessions/{session_id}/messages/", response_model=DiaryMessagesResponse)
async def read_session_messages(
    diary_id: str,
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    messages = list_session_messages(current_user["uid"], diary_id, session_id)
    if messages is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diary session not found")
    return DiaryMessagesResponse(items=messages)
