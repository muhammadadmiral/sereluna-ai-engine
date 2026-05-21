from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from datetime import datetime, timedelta, timezone

from schemas.screening_schema import (
    ScreeningQuestionnaireResponse,
    ScreeningRequest,
    ScreeningResponse,
    ScreeningStatusResponse,
)
from services.context_service import ensure_user_document, save_screening, today_id
from services.firebase_service import get_current_user
from services.nlp_service import score_dass21
from services.screening_service import (
    DASS21_RECOMMENDED_INTERVAL_DAYS,
    get_dass21_questionnaire,
    get_screening_status,
)

router = APIRouter(prefix="/api/v1/screening", tags=["screening"])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@router.get("/dass21/", response_model=ScreeningQuestionnaireResponse)
async def read_dass21_questionnaire():
    return get_dass21_questionnaire()


@router.get("/status/", response_model=ScreeningStatusResponse)
async def read_screening_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return get_screening_status(current_user["uid"])


@router.post("/", response_model=ScreeningResponse)
async def create_screening(
    request: ScreeningRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    uid = current_user["uid"]
    ensure_user_document(uid, current_user)

    try:
        result = score_dass21(request.answers)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    result["answers"] = request.answers
    saved = save_screening(uid, result, request.note or "")
    date_value = saved.get("date") or today_id()
    next_date = datetime.strptime(date_value, "%Y-%m-%d").date() + timedelta(days=DASS21_RECOMMENDED_INTERVAL_DAYS)
    today = datetime.now().astimezone().date()
    updated_at = _utc_now_iso()
    return ScreeningResponse(
        date=date_value,
        scores=result["scores"],
        severity=result["severity"],
        summary=result["summary"],
        algorithm=result.get("algorithm", {}),
        has_screening_today=True,
        next_recommended_date=next_date.isoformat(),
        next_recommended_in_days=max(0, (next_date - today).days),
        recommended_interval_days=DASS21_RECOMMENDED_INTERVAL_DAYS,
        updated_at=updated_at,
        updated_statistics_version=updated_at,
    )
