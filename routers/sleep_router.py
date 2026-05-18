import re
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, status

from schemas.sleep_schema import SleepDailyListResponse, SleepDailyRequest, SleepDailyResponse
from services.firebase_service import get_current_user
from services.sleep_service import list_daily_sleep_metrics, save_daily_sleep_metric

router = APIRouter(prefix="/sleep", tags=["sleep"])
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@router.post("/daily/", response_model=SleepDailyResponse)
async def save_sleep_daily(
    request: SleepDailyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    date_value = request.date.strip()
    if not date_value:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="date is required")
    if not DATE_PATTERN.match(date_value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="date must use yyyy-MM-dd format",
        )
    if request.total_sleep_hours < 0 or request.total_sleep_hours > 24:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="total_sleep_hours must be between 0 and 24",
        )

    save_daily_sleep_metric(
        uid=current_user["uid"],
        date=date_value,
        sleep_quality=request.sleep_quality.strip(),
        total_sleep_hours=request.total_sleep_hours,
    )
    return SleepDailyResponse(success=True)


@router.get("/daily/", response_model=SleepDailyListResponse)
async def read_sleep_daily(
    limit: int = Query(14, ge=1, le=60),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return SleepDailyListResponse(items=list_daily_sleep_metrics(current_user["uid"], limit))
