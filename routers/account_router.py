from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from firebase_admin.auth import UserNotFoundError

from schemas.auth_schema import MessageResponse
from services.account_service import delete_account
from services.firebase_service import get_current_user

router = APIRouter(prefix="/me/account", tags=["account"])


@router.delete("", response_model=MessageResponse)
@router.delete("/", response_model=MessageResponse, include_in_schema=False)
async def delete_my_account(current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        delete_account(current_user["uid"])
    except UserNotFoundError:
        return MessageResponse(message="Account data deleted. Firebase user was already missing.")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account",
        ) from exc

    return MessageResponse(message="Account deleted successfully.")
