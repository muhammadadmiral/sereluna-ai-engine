from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from schemas.screening_schema import ScreeningRequest, ScreeningResponse
from services.context_service import ensure_user_document, save_screening, today_id
from services.firebase_service import get_current_user
from services.nlp_service import score_dass21

router = APIRouter(prefix="/api/v1/screening", tags=["screening"])


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
    return ScreeningResponse(
        date=saved.get("date") or today_id(),
        scores=result["scores"],
        severity=result["severity"],
        summary=result["summary"],
        has_screening_today=True,
    )
