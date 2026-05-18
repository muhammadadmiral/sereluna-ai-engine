from typing import Any, Dict, Optional

from services.firebase_service import serialize_firestore_value, server_timestamp, user_document
from services.notification_service import create_notification


def _provider_from_user(firebase_user: Dict[str, Any], user_data: Dict[str, Any]) -> str:
    firebase_claim = firebase_user.get("claims", {}).get("firebase", {})
    return user_data.get("provider") or firebase_claim.get("sign_in_provider") or ""


def _profile_payload(uid: str, firebase_user: Dict[str, Any], user_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "uid": uid,
        "name": user_data.get("name") or firebase_user.get("name") or "",
        "email": user_data.get("email") or firebase_user.get("email") or "",
        "photo_url": user_data.get("photoUrl") or user_data.get("photo_url") or "",
        "provider": _provider_from_user(firebase_user, user_data),
        "latest_screening_summary": user_data.get("latestScreeningSummary") or "",
        "latest_diary_summary": user_data.get("latestDiarySummary") or "",
        "personal_context": user_data.get("personalContext") or "",
        "has_screening_today": bool(user_data.get("hasScreeningToday", False)),
        "created_at": serialize_firestore_value(user_data.get("createdAt")),
        "updated_at": serialize_firestore_value(user_data.get("updatedAt")),
    }


def get_profile(uid: str, firebase_user: Dict[str, Any]) -> Dict[str, Any]:
    user_ref = user_document(uid)
    snapshot = user_ref.get()
    user_data = snapshot.to_dict() or {}

    if not snapshot.exists:
        user_data = {
            "uid": uid,
            "name": firebase_user.get("name") or "",
            "email": firebase_user.get("email") or "",
            "createdAt": server_timestamp(),
            "updatedAt": server_timestamp(),
        }
        user_ref.set(user_data, merge=True)
        snapshot = user_ref.get()
        user_data = snapshot.to_dict() or user_data

    return _profile_payload(uid, firebase_user, user_data)


def update_profile(
    uid: str,
    firebase_user: Dict[str, Any],
    name: Optional[str] = None,
    photo_url: Optional[str] = None,
) -> Dict[str, Any]:
    user_ref = user_document(uid)
    payload: Dict[str, Any] = {"updatedAt": server_timestamp()}

    if name is not None:
        payload["name"] = name.strip()
    if photo_url is not None:
        payload["photoUrl"] = photo_url.strip()

    user_ref.set(payload, merge=True)
    create_notification(uid, "Profil diperbarui", "Data profil kamu berhasil diperbarui.", "profile")
    return get_profile(uid, firebase_user)
