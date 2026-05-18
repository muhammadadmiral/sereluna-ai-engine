from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from schemas.device_schema import DeviceTokenRequest, DeviceTokenResponse
from services.device_service import save_device_token
from services.firebase_service import get_current_user

router = APIRouter(prefix="/device-token", tags=["device"])


@router.post("/", response_model=DeviceTokenResponse)
async def register_device_token(
    request: DeviceTokenRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    token = request.token.strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="token is required")

    save_device_token(current_user["uid"], token)
    return DeviceTokenResponse(success=True)
