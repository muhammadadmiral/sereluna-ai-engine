from typing import Any, Dict, List, Optional

from firebase_admin import firestore
from google.cloud.firestore_v1 import FieldFilter

from services.firebase_service import get_firestore_client, serialize_firestore_value, server_timestamp, user_document


def create_notification(
    uid: str,
    title: str,
    body: str,
    notification_type: str = "system",
    action_link: Optional[str] = None,
) -> str:
    _update_time, notification_ref = user_document(uid).collection("notifications").add(
        {
            "title": title,
            "body": body,
            "type": notification_type,
            "isRead": False,
            "actionLink": action_link,
            "createdAt": server_timestamp(),
        }
    )
    return notification_ref.id


def list_notifications(uid: str, limit: int = 30) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(limit, 100))
    notifications_ref = user_document(uid).collection("notifications")
    items: List[Dict[str, Any]] = []
    for snapshot in (
        notifications_ref.order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(safe_limit)
        .stream()
    ):
        data = snapshot.to_dict() or {}
        items.append(
            {
                "id": snapshot.id,
                "title": data.get("title") or "",
                "body": data.get("body") or "",
                "type": data.get("type") or "",
                "is_read": bool(data.get("isRead", data.get("is_read", False))),
                "created_at": serialize_firestore_value(data.get("createdAt") or data.get("created_at")),
                "action_link": data.get("actionLink") or data.get("action_link"),
            }
        )
    return items


def mark_notification_read(uid: str, notification_id: str) -> bool:
    notification_ref = user_document(uid).collection("notifications").document(notification_id)
    snapshot = notification_ref.get()
    if not snapshot.exists:
        return False

    notification_ref.set({"isRead": True, "updatedAt": server_timestamp()}, merge=True)
    return True


def mark_all_notifications_read(uid: str) -> int:
    notifications_ref = user_document(uid).collection("notifications")
    batch = get_firestore_client().batch()
    updated = 0

    for snapshot in notifications_ref.where(filter=FieldFilter("isRead", "==", False)).limit(500).stream():
        batch.set(snapshot.reference, {"isRead": True, "updatedAt": server_timestamp()}, merge=True)
        updated += 1

    if updated:
        batch.commit()
    return updated
