from typing import Any, Dict

from fastapi import APIRouter, Depends, Query

from schemas.stats_schema import MoodDistributionResponse, SleepTrendResponse
from services.firebase_service import get_current_user
from services.stats_service import get_mood_distribution, get_sleep_trends

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/mood-distribution/", response_model=MoodDistributionResponse)
@router.get("/mood-distribution", response_model=MoodDistributionResponse, include_in_schema=False)
async def read_mood_distribution(
    days: int = Query(7, ge=1, le=90),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return get_mood_distribution(uid=current_user["uid"], days=days)


@router.get("/sleep-trends/", response_model=SleepTrendResponse)
@router.get("/sleep-trends", response_model=SleepTrendResponse, include_in_schema=False)
async def read_sleep_trends(
    days: int = Query(7, ge=1, le=90),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return get_sleep_trends(uid=current_user["uid"], days=days)
