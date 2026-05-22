from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from schemas.profile_schema import ProfilePhotoUpdateRequest, ProfileResponse, ProfileUpdateRequest
from services.firebase_service import get_current_user
from services.media_service import save_profile_photo
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


@router.patch("/photo/", response_model=ProfileResponse)
async def patch_profile_photo(
    request: ProfilePhotoUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return ProfileResponse(
        **update_profile(
            uid=current_user["uid"],
            firebase_user=current_user,
            photo_url=request.photo_url,
        )
    )


@router.post("/photo/", response_model=ProfileResponse)
async def upload_profile_photo(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        content = await file.read()
        save_profile_photo(
            uid=current_user["uid"],
            content=content,
            content_type=file.content_type or "",
            original_name=file.filename or "",
        )
        return ProfileResponse(**get_profile(current_user["uid"], current_user))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
