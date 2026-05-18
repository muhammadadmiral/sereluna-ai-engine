from typing import Any, Dict

from fastapi import APIRouter, Depends

from schemas.chat_schema import UserContextResponse
from services.context_service import ensure_user_document, get_user_context
from services.firebase_service import get_current_user

router = APIRouter(prefix="/api/v1/me", tags=["context"])


@router.get("/context/", response_model=UserContextResponse)
async def get_context(current_user: Dict[str, Any] = Depends(get_current_user)):
    uid = current_user["uid"]
    ensure_user_document(uid, current_user)
    return UserContextResponse(**get_user_context(uid))
