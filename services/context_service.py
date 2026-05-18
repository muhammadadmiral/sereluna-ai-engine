import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from firebase_admin import firestore

from services.firebase_service import get_firestore_client


APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Jakarta")


def today_id() -> str:
    return datetime.now(ZoneInfo(APP_TIMEZONE)).strftime("%Y-%m-%d")


def _server_timestamp():
    return firestore.SERVER_TIMESTAMP


def _user_ref(uid: str):
    return get_firestore_client().collection("users").document(uid)


def ensure_user_document(uid: str, firebase_user: Dict[str, Any]) -> Dict[str, Any]:
    user_ref = _user_ref(uid)
    snapshot = user_ref.get()
    existing = snapshot.to_dict() or {}

    profile_data = {
        "uid": uid,
        "email": firebase_user.get("email") or existing.get("email", ""),
        "name": firebase_user.get("name") or existing.get("name", "Teman"),
        "updatedAt": _server_timestamp(),
    }
    if not snapshot.exists:
        profile_data["createdAt"] = _server_timestamp()

    user_ref.set(profile_data, merge=True)
    existing.update(profile_data)
    return existing


def get_or_create_today_diary(uid: str, room_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    diary_id = room_id or today_id()
    diary_ref = _user_ref(uid).collection("diaries").document(diary_id)
    snapshot = diary_ref.get()
    if not snapshot.exists:
        diary_ref.set(
            {
                "date": diary_id,
                "chatSummary": "",
                "createdAt": _server_timestamp(),
                "updatedAt": _server_timestamp(),
            },
            merge=True,
        )
        return diary_id, {"date": diary_id, "chatSummary": ""}

    return diary_id, snapshot.to_dict() or {}


def get_or_create_session(uid: str, diary_id: str, session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    sessions_ref = _user_ref(uid).collection("diaries").document(diary_id).collection("sessions")

    if session_id:
        session_ref = sessions_ref.document(session_id)
        snapshot = session_ref.get()
        if not snapshot.exists:
            session_ref.set(
                {
                    "summary": "",
                    "status": "active",
                    "createdAt": _server_timestamp(),
                    "updatedAt": _server_timestamp(),
                },
                merge=True,
            )
            return session_id, {"summary": "", "status": "active"}
        return session_id, snapshot.to_dict() or {}

    active_sessions = sessions_ref.where("status", "==", "active").limit(1).stream()
    for session_snapshot in active_sessions:
        return session_snapshot.id, session_snapshot.to_dict() or {}

    session_ref = sessions_ref.document()
    session_ref.set(
        {
            "summary": "",
            "status": "active",
            "createdAt": _server_timestamp(),
            "updatedAt": _server_timestamp(),
        },
        merge=True,
    )
    return session_ref.id, {"summary": "", "status": "active"}


def save_message(
    uid: str,
    diary_id: str,
    session_id: str,
    role: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    messages_ref = (
        _user_ref(uid)
        .collection("diaries")
        .document(diary_id)
        .collection("sessions")
        .document(session_id)
        .collection("messages")
    )
    _update_time, message_ref = messages_ref.add(
        {
            "role": role,
            "text": text,
            "content": text,
            "metadata": metadata or {},
            "createdAt": _server_timestamp(),
        }
    )
    return message_ref.id


def get_session_messages(uid: str, diary_id: str, session_id: str) -> List[Dict[str, Any]]:
    messages_ref = (
        _user_ref(uid)
        .collection("diaries")
        .document(diary_id)
        .collection("sessions")
        .document(session_id)
        .collection("messages")
    )
    messages: List[Dict[str, Any]] = []
    for snapshot in messages_ref.order_by("createdAt").stream():
        data = snapshot.to_dict() or {}
        data["id"] = snapshot.id
        messages.append(data)
    return messages


def format_messages(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = "User" if message.get("role") == "user" else "Sereluna"
        text = message.get("text") or message.get("content") or ""
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def build_memory_context(
    profile_context: str,
    latest_screening_summary: str,
    latest_diary_summary: str,
    session_summary: str,
    history_text: str,
    past_diaries: List[str],
) -> str:
    sections: List[str] = []
    if profile_context.strip():
        sections.append(f"Profil inti:\n{profile_context.strip()}")
    if latest_screening_summary.strip():
        sections.append(f"Ringkasan screening terbaru:\n{latest_screening_summary.strip()}")
    if latest_diary_summary.strip():
        sections.append(f"Ringkasan diary terbaru:\n{latest_diary_summary.strip()}")
    if session_summary.strip():
        sections.append(f"Ringkasan sesi berjalan:\n{session_summary.strip()}")
    if past_diaries:
        recent_diaries = "\n".join(f"- {item}" for item in past_diaries[:5] if item and item.strip())
        if recent_diaries.strip():
            sections.append(f"Diary relevan sebelumnya:\n{recent_diaries}")
    if history_text.strip():
        sections.append(f"Transcript chat terbaru:\n{history_text.strip()}")
    return "\n\n".join(sections).strip()


def get_recent_diary_summaries(uid: str, limit: int = 5) -> List[str]:
    diaries_ref = _user_ref(uid).collection("diaries")
    summaries: List[str] = []
    for snapshot in diaries_ref.order_by("updatedAt", direction=firestore.Query.DESCENDING).limit(limit).stream():
        data = snapshot.to_dict() or {}
        summary = data.get("chatSummary") or data.get("summary") or ""
        if summary:
            summaries.append(summary)
    return summaries


def get_chat_context(uid: str, diary_id: str, session_id: str) -> Dict[str, Any]:
    user_snapshot = _user_ref(uid).get()
    user_data = user_snapshot.to_dict() or {}

    session_snapshot = (
        _user_ref(uid)
        .collection("diaries")
        .document(diary_id)
        .collection("sessions")
        .document(session_id)
        .get()
    )
    session_data = session_snapshot.to_dict() or {}

    messages = get_session_messages(uid, diary_id, session_id)
    name = user_data.get("name") or "Teman"
    email = user_data.get("email") or ""
    profile_context = f"Nama: {name}. Email: {email or 'N/A'}."
    if user_data.get("personalContext"):
        profile_context = f"{profile_context} Personal context: {user_data.get('personalContext')}"

    latest_screening_summary = user_data.get("latestScreeningSummary") or ""
    latest_diary_summary = user_data.get("latestDiarySummary") or ""
    past_diaries = get_recent_diary_summaries(uid, limit=5)
    has_screening_today = bool(user_data.get("hasScreeningToday")) and user_data.get("lastScreeningDate") == today_id()
    recent_history_text = format_messages(messages[-60:])
    memory_context = build_memory_context(
        profile_context=profile_context,
        latest_screening_summary=latest_screening_summary,
        latest_diary_summary=latest_diary_summary,
        session_summary=session_data.get("summary") or latest_diary_summary,
        history_text=recent_history_text,
        past_diaries=past_diaries,
    )

    return {
        "name": name,
        "email": email,
        "profile_context": profile_context,
        "latest_screening_summary": latest_screening_summary,
        "latest_diary_summary": latest_diary_summary,
        "session_summary": session_data.get("summary") or latest_diary_summary,
        "session_history": recent_history_text,
        "memory_context": memory_context,
        "past_diaries": past_diaries,
        "has_screening_today": has_screening_today,
    }


def update_chat_summaries(uid: str, diary_id: str, session_id: str, summary: str) -> None:
    user_ref = _user_ref(uid)
    diary_ref = user_ref.collection("diaries").document(diary_id)
    session_ref = diary_ref.collection("sessions").document(session_id)

    session_ref.set({"summary": summary, "updatedAt": _server_timestamp()}, merge=True)
    diary_ref.set({"chatSummary": summary, "updatedAt": _server_timestamp()}, merge=True)
    user_ref.set(
        {
            "latestDiarySummary": summary,
            "personalContext": summary,
            "updatedAt": _server_timestamp(),
        },
        merge=True,
    )


def finish_session(uid: str, diary_id: str, session_id: str, final_summary: str) -> None:
    user_ref = _user_ref(uid)
    diary_ref = user_ref.collection("diaries").document(diary_id)
    session_ref = diary_ref.collection("sessions").document(session_id)

    session_ref.set(
        {
            "summary": final_summary,
            "status": "finished",
            "endTime": _server_timestamp(),
            "updatedAt": _server_timestamp(),
        },
        merge=True,
    )
    diary_ref.set({"chatSummary": final_summary, "updatedAt": _server_timestamp()}, merge=True)
    user_ref.set(
        {
            "latestDiarySummary": final_summary,
            "personalContext": final_summary,
            "updatedAt": _server_timestamp(),
        },
        merge=True,
    )
    user_ref.collection("personalContexts").add(
        {
            "type": "diary_summary",
            "source": "chat_finish",
            "diaryId": diary_id,
            "sessionId": session_id,
            "summary": final_summary,
            "createdAt": _server_timestamp(),
        }
    )


def save_screening(uid: str, result: Dict[str, Any], note: str = "") -> Dict[str, Any]:
    date_id = today_id()
    user_ref = _user_ref(uid)
    payload = {
        "date": date_id,
        "answers": result["answers"],
        "scores": result["scores"],
        "severity": result["severity"],
        "summary": result["summary"],
        "algorithm": result.get("algorithm", {}),
        "note": note or "",
        "createdAt": _server_timestamp(),
        "updatedAt": _server_timestamp(),
    }

    user_ref.collection("screenings").document(date_id).set(payload, merge=True)
    user_ref.collection("medicalRecords").document(date_id).set(payload, merge=True)
    user_ref.set(
        {
            "latestScreeningSummary": result["summary"],
            "hasScreeningToday": True,
            "lastScreeningDate": date_id,
            "updatedAt": _server_timestamp(),
        },
        merge=True,
    )
    return payload


def get_user_context(uid: str) -> Dict[str, Any]:
    user_snapshot = _user_ref(uid).get()
    user_data = user_snapshot.to_dict() or {}
    name = user_data.get("name") or "Teman"
    email = user_data.get("email") or ""
    personal_context = user_data.get("personalContext") or ""
    profile_context = f"Nama: {name}. Email: {email or 'N/A'}."
    if personal_context:
        profile_context = f"{profile_context} Personal context: {personal_context}"

    return {
        "profile_context": profile_context,
        "latest_screening_summary": user_data.get("latestScreeningSummary") or "",
        "latest_diary_summary": user_data.get("latestDiarySummary") or "",
        "past_diaries": get_recent_diary_summaries(uid, limit=5),
        "has_screening_today": bool(user_data.get("hasScreeningToday"))
        and user_data.get("lastScreeningDate") == today_id(),
    }
