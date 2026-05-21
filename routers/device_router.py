from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from schemas.device_schema import DeviceTokenRequest, DeviceTokenResponse
from services.device_service import save_device_token
from services.firebase_service import get_current_user
from services.notification_service import create_notification

router = APIRouter(prefix="/device-token", tags=["device"])


@router.post("/", response_model=DeviceTokenResponse)
async def register_device_token(
    request: DeviceTokenRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    token = request.token.strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="token is required")

    is_new_token = save_device_token(current_user["uid"], token)
    if is_new_token:
        create_notification(
            uid=current_user["uid"],
            title="Perangkat tersambung",
            body="Akun Sereluna kamu baru saja tersambung di perangkat ini.",
            notification_type="system",
            priority="low",
            category_label="Keamanan",
            action_link="/settings/security",
            notification_key=f"device_token:{token[-16:]}",
        )
    return DeviceTokenResponse(success=True)
