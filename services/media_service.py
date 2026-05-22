import base64
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from google.cloud import firestore

from services.firebase_service import get_storage_bucket, server_timestamp, user_document
from services.llm_service import analyze_image_with_nvidia


ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _media_collection(uid: str):
    return user_document(uid).collection("media")


def _signed_url(blob, minutes: int = 60) -> Optional[str]:
    try:
        return blob.generate_signed_url(
            expiration=timedelta(minutes=minutes),
            method="GET",
            version="v4",
        )
    except Exception:
        return None


def save_user_image(uid: str, content: bytes, content_type: str, original_name: str = "") -> Dict[str, Any]:
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError("Unsupported image type")
    if not content:
        raise ValueError("Image file is empty")
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image file is too large")

    media_id = uuid.uuid4().hex
    extension = ALLOWED_IMAGE_TYPES[content_type]
    safe_name = Path(original_name or f"upload{extension}").name
    storage_path = f"users/{uid}/media/{media_id}{extension}"

    bucket = get_storage_bucket()
    blob = bucket.blob(storage_path)
    blob.upload_from_string(content, content_type=content_type)

    doc = {
        "media_id": media_id,
        "storage_path": storage_path,
        "original_name": safe_name,
        "content_type": content_type,
        "size_bytes": len(content),
        "created_at": server_timestamp(),
        "updated_at": server_timestamp(),
    }
    _media_collection(uid).document(media_id).set(doc)

    return {
        "media_id": media_id,
        "storage_path": storage_path,
        "content_type": content_type,
        "size_bytes": len(content),
        "signed_url": _signed_url(blob),
        "updated_at": _utc_now_iso(),
    }


def save_profile_photo(uid: str, content: bytes, content_type: str, original_name: str = "") -> Dict[str, Any]:
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError("Unsupported image type")
    if not content:
        raise ValueError("Image file is empty")
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image file is too large")

    photo_id = uuid.uuid4().hex
    extension = ALLOWED_IMAGE_TYPES[content_type]
    safe_name = Path(original_name or f"profile{extension}").name
    storage_path = f"users/{uid}/profile/{photo_id}{extension}"
    download_token = uuid.uuid4().hex

    bucket = get_storage_bucket()
    blob = bucket.blob(storage_path)
    blob.metadata = {"firebaseStorageDownloadTokens": download_token}
    blob.upload_from_string(content, content_type=content_type)
    blob.patch()

    photo_url = (
        f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/"
        f"{quote(storage_path, safe='')}?alt=media&token={download_token}"
    )
    user_document(uid).set(
        {
            "photoUrl": photo_url,
            "photoStoragePath": storage_path,
            "photoOriginalName": safe_name,
            "updatedAt": server_timestamp(),
        },
        merge=True,
    )
    return {
        "photo_url": photo_url,
        "storage_path": storage_path,
        "content_type": content_type,
        "size_bytes": len(content),
        "updated_at": _utc_now_iso(),
    }


def analyze_user_image(uid: str, media_id: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    snapshot = _media_collection(uid).document(media_id).get()
    if not snapshot.exists:
        raise ValueError("Media not found")

    data = snapshot.to_dict() or {}
    content_type = data.get("content_type")
    storage_path = data.get("storage_path")
    if content_type not in ALLOWED_IMAGE_TYPES or not storage_path:
        raise ValueError("Media is not an analyzable image")

    bucket = get_storage_bucket()
    blob = bucket.blob(storage_path)
    content = blob.download_as_bytes()
    data_url = f"data:{content_type};base64,{base64.b64encode(content).decode('ascii')}"

    result = analyze_image_with_nvidia(data_url, prompt or "")
    _media_collection(uid).document(media_id).set(
        {
            "last_analysis": result,
            "updated_at": server_timestamp(),
            "analyzed_at": server_timestamp(),
        },
        merge=True,
    )

    return {
        "media_id": media_id,
        "model": result["model"],
        "analysis": result["analysis"],
        "updated_at": _utc_now_iso(),
    }


def analyze_images_for_chat(uid: str, media_ids: list[str]) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    unique_ids = []
    for media_id in media_ids or []:
        clean_id = str(media_id or "").strip()
        if clean_id and clean_id not in unique_ids:
            unique_ids.append(clean_id)

    for media_id in unique_ids[:3]:
        snapshot = _media_collection(uid).document(media_id).get()
        if not snapshot.exists:
            results.append({"media_id": media_id, "error": "not_found"})
            continue
        data = snapshot.to_dict() or {}
        cached = data.get("last_analysis")
        if isinstance(cached, dict) and cached.get("analysis"):
            results.append(
                {
                    "media_id": media_id,
                    "model": cached.get("model", ""),
                    "analysis": cached.get("analysis", ""),
                    "cached": True,
                }
            )
            continue
        try:
            result = analyze_user_image(
                uid=uid,
                media_id=media_id,
                prompt="Ini gambar yang dilampirkan user ke chat. Jika screenshot chat, rangkum konteks, nada emosi, dan hal penting yang perlu diperhatikan.",
            )
            result["cached"] = False
            results.append(result)
        except Exception as exc:
            results.append({"media_id": media_id, "error": str(exc)})
    return results
