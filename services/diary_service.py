from typing import Any, Dict, List, Optional

from firebase_admin import firestore

from services.firebase_service import serialize_firestore_value, user_document


def _not_found_none(snapshot) -> Optional[Dict[str, Any]]:
    if not snapshot.exists:
        return None
    return snapshot.to_dict() or {}


def list_diaries(uid: str, limit: int = 30) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(limit, 100))
    diaries_ref = user_document(uid).collection("diaries")
    items: List[Dict[str, Any]] = []
    for snapshot in diaries_ref.order_by("date", direction=firestore.Query.DESCENDING).limit(safe_limit).stream():
        data = snapshot.to_dict() or {}
        items.append(
            {
                "id": snapshot.id,
                "date": data.get("date") or snapshot.id,
                "chat_summary": data.get("chatSummary") or data.get("chat_summary") or "",
                "created_at": serialize_firestore_value(data.get("createdAt") or data.get("created_at")),
                "updated_at": serialize_firestore_value(data.get("updatedAt") or data.get("updated_at")),
            }
        )
    return items


def get_diary_detail(uid: str, diary_id: str) -> Optional[Dict[str, Any]]:
    diary_ref = user_document(uid).collection("diaries").document(diary_id)
    diary_data = _not_found_none(diary_ref.get())
    if diary_data is None:
        return None

    sessions: List[Dict[str, Any]] = []
    for snapshot in diary_ref.collection("sessions").order_by("createdAt").stream():
        data = snapshot.to_dict() or {}
        sessions.append(
            {
                "id": snapshot.id,
                "model": data.get("model") or data.get("llmModel") or "",
                "summary": data.get("summary") or "",
                "start_time": serialize_firestore_value(
                    data.get("startTime") or data.get("createdAt") or data.get("start_time")
                ),
                "end_time": serialize_firestore_value(data.get("endTime") or data.get("end_time")),
            }
        )

    return {
        "id": diary_id,
        "date": diary_data.get("date") or diary_id,
        "chat_summary": diary_data.get("chatSummary") or diary_data.get("chat_summary") or "",
        "sessions": sessions,
    }


def list_session_messages(uid: str, diary_id: str, session_id: str) -> Optional[List[Dict[str, Any]]]:
    diary_ref = user_document(uid).collection("diaries").document(diary_id)
    session_ref = diary_ref.collection("sessions").document(session_id)
    if not diary_ref.get().exists or not session_ref.get().exists:
        return None

    items: List[Dict[str, Any]] = []
    for snapshot in session_ref.collection("messages").order_by("createdAt").stream():
        data = snapshot.to_dict() or {}
        items.append(
            {
                "id": snapshot.id,
                "sender_role": data.get("senderRole") or data.get("role") or "",
                "text": data.get("text") or data.get("content") or "",
                "timestamp": serialize_firestore_value(data.get("timestamp") or data.get("createdAt")),
            }
        )
    return items
