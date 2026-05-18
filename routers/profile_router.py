from typing import Any, Dict

from fastapi import APIRouter, Depends

from schemas.profile_schema import ProfileResponse, ProfileUpdateRequest
from services.firebase_service import get_current_user
from services.profile_service import get_profile, update_profile

router = APIRouter(prefix="/me/profile", tags=["profile"])


@router.get("/", response_model=ProfileResponse)
async def read_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    return ProfileResponse(**get_profile(current_user["uid"], current_user))


@router.put("/", response_model=ProfileResponse)
async def update_my_profile(
    request: ProfileUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return ProfileResponse(
        **update_profile(
            uid=current_user["uid"],
            firebase_user=current_user,
            name=request.name,
            photo_url=request.photo_url,
        )
    )
