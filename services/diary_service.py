from typing import Any, Dict, List, Optional

from firebase_admin import firestore

from services.firebase_service import serialize_firestore_value, user_document
from services.summary_service import clean_diary_summary


PREVIEW_LENGTH = 160


def _not_found_none(snapshot) -> Optional[Dict[str, Any]]:
    if not snapshot.exists:
        return None
    return snapshot.to_dict() or {}


def _preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _session_item(snapshot) -> Dict[str, Any]:
    data = snapshot.to_dict() or {}
    summary = clean_diary_summary(data.get("summary") or "")
    return {
        "id": snapshot.id,
        "model": data.get("model") or data.get("llmModel") or "",
        "summary": summary,
        "preview": _preview(summary),
        "status": data.get("status") or "",
        "start_time": serialize_firestore_value(
            data.get("startTime") or data.get("createdAt") or data.get("start_time")
        ),
        "end_time": serialize_firestore_value(data.get("endTime") or data.get("end_time")),
        "updated_at": serialize_firestore_value(data.get("updatedAt") or data.get("updated_at")),
    }


def _list_sessions(diary_ref) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    for snapshot in diary_ref.collection("sessions").order_by("createdAt").stream():
        sessions.append(_session_item(snapshot))
    return sessions


def _diary_item(snapshot) -> Dict[str, Any]:
    data = snapshot.to_dict() or {}
    chat_summary = clean_diary_summary(data.get("chatSummary") or data.get("chat_summary") or "")
    sessions = _list_sessions(snapshot.reference)
    return {
        "id": snapshot.id,
        "date": data.get("date") or snapshot.id,
        "chat_summary": chat_summary,
        "preview": _preview(chat_summary),
        "session_count": len(sessions),
        "sessions": sessions,
        "created_at": serialize_firestore_value(data.get("createdAt") or data.get("created_at")),
        "updated_at": serialize_firestore_value(data.get("updatedAt") or data.get("updated_at")),
    }


def list_diaries(uid: str, limit: int = 30) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(limit, 100))
    diaries_ref = user_document(uid).collection("diaries")
    items: List[Dict[str, Any]] = []
    for snapshot in diaries_ref.order_by("date", direction=firestore.Query.DESCENDING).limit(safe_limit).stream():
        items.append(_diary_item(snapshot))
    return items


def list_diary_entries(uid: str, limit: int = 30) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(limit, 100))
    diaries_ref = user_document(uid).collection("diaries")
    entries: List[Dict[str, Any]] = []

    for diary_snapshot in diaries_ref.order_by("date", direction=firestore.Query.DESCENDING).limit(100).stream():
        diary_data = diary_snapshot.to_dict() or {}
        diary_id = diary_snapshot.id
        date = diary_data.get("date") or diary_id

        for session in _list_sessions(diary_snapshot.reference):
            session_id = session["id"]
            entries.append(
                {
                    "id": f"{diary_id}:{session_id}",
                    "diary_id": diary_id,
                    "session_id": session_id,
                    "date": date,
                    "summary": session["summary"],
                    "preview": session["preview"],
                    "status": session["status"],
                    "model": session["model"],
                    "start_time": session["start_time"],
                    "end_time": session["end_time"],
                    "updated_at": session["updated_at"],
                }
            )

    entries.sort(
        key=lambda item: (
            item.get("date") or "",
            item.get("updated_at") or item.get("start_time") or "",
            item.get("session_id") or "",
        ),
        reverse=True,
    )
    return entries[:safe_limit]


def get_diary_detail(uid: str, diary_id: str) -> Optional[Dict[str, Any]]:
    diary_ref = user_document(uid).collection("diaries").document(diary_id)
    diary_data = _not_found_none(diary_ref.get())
    if diary_data is None:
        return None

    chat_summary = clean_diary_summary(diary_data.get("chatSummary") or diary_data.get("chat_summary") or "")
    sessions = _list_sessions(diary_ref)

    return {
        "id": diary_id,
        "date": diary_data.get("date") or diary_id,
        "chat_summary": chat_summary,
        "preview": _preview(chat_summary),
        "session_count": len(sessions),
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
