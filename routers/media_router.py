from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from schemas.media_schema import ImageAnalysisRequest, ImageAnalysisResponse, MediaUploadResponse
from services.firebase_service import get_current_user
from services.media_service import analyze_user_image, save_user_image


router = APIRouter(prefix="/media", tags=["media"])


@router.post("/images/", response_model=MediaUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        content = await file.read()
        return save_user_image(
            uid=current_user["uid"],
            content=content,
            content_type=file.content_type or "",
            original_name=file.filename or "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/images/analyze/", response_model=ImageAnalysisResponse)
async def analyze_image(
    request: ImageAnalysisRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        return analyze_user_image(
            uid=current_user["uid"],
            media_id=request.media_id,
            prompt=request.prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
