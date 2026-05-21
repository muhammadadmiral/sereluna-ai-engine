from typing import Any, Dict

from fastapi import APIRouter, Depends, Query

from schemas.stats_schema import WellbeingStatisticsResponse
from services.firebase_service import get_current_user
from services.stats_service import get_wellbeing_statistics

router = APIRouter(prefix="/statistics", tags=["statistics"])


@router.get("/wellbeing/", response_model=WellbeingStatisticsResponse)
@router.get("/wellbeing", response_model=WellbeingStatisticsResponse, include_in_schema=False)
async def read_wellbeing_statistics(
    range: str = Query("30d", pattern="^(7d|30d|90d)$"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return get_wellbeing_statistics(uid=current_user["uid"], range_value=range)
