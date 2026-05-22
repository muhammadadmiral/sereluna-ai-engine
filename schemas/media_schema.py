from typing import Optional

from pydantic import BaseModel


class MediaUploadResponse(BaseModel):
    media_id: str
    storage_path: str
    content_type: str
    size_bytes: int
    signed_url: Optional[str] = None
    updated_at: str


class ImageAnalysisRequest(BaseModel):
    media_id: str
    prompt: Optional[str] = None


class ImageAnalysisResponse(BaseModel):
    media_id: str
    model: str
    analysis: str
    updated_at: str
