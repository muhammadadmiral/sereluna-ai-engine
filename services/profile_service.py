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


def _base_profile_data(uid: str, firebase_user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "uid": uid,
        "email": firebase_user.get("email") or "",
        "name": firebase_user.get("name") or "",
        "provider": _provider_from_user(firebase_user, {}),
        "latestScreeningSummary": "",
        "latestDiarySummary": "",
        "personalContext": "",
        "hasScreeningToday": False,
    }


def get_profile(uid: str, firebase_user: Dict[str, Any]) -> Dict[str, Any]:
    user_ref = user_document(uid)
    snapshot = user_ref.get()
    user_data = snapshot.to_dict() or {}

    if not snapshot.exists:
        user_data = {
            **_base_profile_data(uid, firebase_user),
            "createdAt": server_timestamp(),
            "updatedAt": server_timestamp(),
        }
        user_ref.set(serialize_firestore_value(user_data), merge=True)
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
    snapshot = user_ref.get()
    existing = snapshot.to_dict() or {}

    payload: Dict[str, Any] = {
        "uid": uid,
        "email": existing.get("email") or firebase_user.get("email") or "",
        "provider": existing.get("provider") or _provider_from_user(firebase_user, existing),
        "updatedAt": server_timestamp(),
    }
    if not snapshot.exists:
        payload.update(_base_profile_data(uid, firebase_user))
        payload["createdAt"] = server_timestamp()

    if name is not None:
        payload["name"] = name.strip()
    elif not snapshot.exists:
        payload["name"] = firebase_user.get("name") or ""

    if photo_url is not None:
        payload["photoUrl"] = photo_url.strip()
    elif not snapshot.exists:
        payload["photoUrl"] = ""

    user_ref.set(serialize_firestore_value(payload), merge=True)
    create_notification(uid, "Profil diperbarui", "Data profil kamu berhasil diperbarui.", "profile")
    return get_profile(uid, firebase_user)
