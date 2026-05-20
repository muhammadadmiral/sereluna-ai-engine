import re
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, status

from schemas.calendar_schema import CalendarDetailResponse, CalendarSummaryResponse
from services.daily_dashboard_service import get_calendar_detail, list_calendar_summary
from services.firebase_service import get_current_user

router = APIRouter(prefix="/calendar", tags=["calendar"])
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@router.get("/summary/", response_model=CalendarSummaryResponse)
async def read_calendar_summary(
    year: int = Query(..., ge=2000, le=2100),
    month: int = Query(..., ge=1, le=12),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return CalendarSummaryResponse(items=list_calendar_summary(current_user["uid"], year, month))


@router.get("/detail/", response_model=CalendarDetailResponse)
async def read_calendar_detail(
    date: str = Query(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    date_value = date.strip()
    if not DATE_PATTERN.match(date_value):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="date must use yyyy-MM-dd format")
    return CalendarDetailResponse(**get_calendar_detail(current_user["uid"], date_value))
