import calendar
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from firebase_admin import firestore

from services.firebase_service import server_timestamp, user_document
from services.summary_service import clean_diary_summary


APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Jakarta")
MOOD_VALUES = {"happy", "neutral", "sad", "anxious", "angry"}
DIARY_SNIPPET_LENGTH = 50
CONTEXT_SNIPPET_LENGTH = 90


def _daily_metrics_collection(uid: str):
    return user_document(uid).collection("analytics").document("dailyMetrics").collection("dates")


def _diaries_collection(uid: str):
    return user_document(uid).collection("diaries")


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def _snippet(value: str, limit: int) -> str:
    normalized = _normalize_text(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _metric_sleep_payload(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sleep_quality = data.get("sleepQuality") or data.get("sleep_quality") or ""
    total_sleep_hours = data.get("totalSleepHours", data.get("total_sleep_hours"))
    bedtime = data.get("bedtime") or data.get("bedtimeAt")
    wakeup = data.get("wakeup") or data.get("wakeupAt")

    has_sleep_data = total_sleep_hours is not None or bool(sleep_quality or bedtime or wakeup)
    if not has_sleep_data:
        return None

    try:
        sleep_hours = float(total_sleep_hours or 0)
    except (TypeError, ValueError):
        sleep_hours = 0

    return {
        "total_sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "bedtime": bedtime,
        "wakeup": wakeup,
    }


def _metric_mood(data: Dict[str, Any]) -> Optional[str]:
    mood = str(data.get("mood") or "").strip().lower()
    return mood if mood in MOOD_VALUES else None


def _diary_summary_from_snapshot(snapshot) -> str:
    if not snapshot.exists:
        return ""

    data = snapshot.to_dict() or {}
    summary = clean_diary_summary(data.get("chatSummary") or data.get("chat_summary") or data.get("summary") or "")
    if summary:
        return summary

    for session_snapshot in snapshot.reference.collection("sessions").order_by(
        "updatedAt", direction=firestore.Query.DESCENDING
    ).limit(1).stream():
        session_data = session_snapshot.to_dict() or {}
        summary = clean_diary_summary(session_data.get("summary") or "")
        if summary:
            return summary

    return ""


def _diary_summary_for_date(uid: str, date_value: str) -> str:
    diary_ref = _diaries_collection(uid).document(date_value)
    summary = _diary_summary_from_snapshot(diary_ref.get())
    if summary:
        return summary

    for snapshot in _diaries_collection(uid).where("date", "==", date_value).limit(1).stream():
        summary = _diary_summary_from_snapshot(snapshot)
        if summary:
            return summary

    return ""


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


def _recent_date_ids(days: int = 3) -> List[str]:
    today = datetime.now(ZoneInfo(APP_TIMEZONE)).date()
    return [(today - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(max(1, days))]


def save_daily_mood(uid: str, date: str, mood: str) -> None:
    _daily_metrics_collection(uid).document(date).set(
        {
            "date": date,
            "mood": mood,
            "updatedAt": server_timestamp(),
        },
        merge=True,
    )


def list_calendar_summary(uid: str, year: int, month: int) -> List[Dict[str, Any]]:
    start_date, end_date = _month_bounds(year, month)
    items_by_date: Dict[str, Dict[str, Any]] = {}

    metrics_ref = _daily_metrics_collection(uid)
    for snapshot in metrics_ref.where("date", ">=", start_date).where("date", "<=", end_date).stream():
        data = snapshot.to_dict() or {}
        date_value = data.get("date") or snapshot.id
        if not start_date <= date_value <= end_date:
            continue

        sleep_payload = _metric_sleep_payload(data)
        items_by_date[date_value] = {
            "date": date_value,
            "has_sleep_data": sleep_payload is not None,
            "mood": _metric_mood(data),
            "has_diary": False,
        }

    diaries_ref = _diaries_collection(uid)
    for snapshot in diaries_ref.where("date", ">=", start_date).where("date", "<=", end_date).stream():
        data = snapshot.to_dict() or {}
        date_value = data.get("date") or snapshot.id
        if not start_date <= date_value <= end_date:
            continue

        item = items_by_date.setdefault(
            date_value,
            {
                "date": date_value,
                "has_sleep_data": False,
                "mood": None,
                "has_diary": False,
            },
        )
        item["has_diary"] = True

    return [items_by_date[date_value] for date_value in sorted(items_by_date)]


def get_calendar_detail(uid: str, date: str) -> Dict[str, Any]:
    metric_snapshot = _daily_metrics_collection(uid).document(date).get()
    metric_data = metric_snapshot.to_dict() or {}
    diary_summary = _diary_summary_for_date(uid, date)

    return {
        "date": date,
        "mood": _metric_mood(metric_data),
        "sleep": _metric_sleep_payload(metric_data),
        "diary_snippet": _snippet(diary_summary, DIARY_SNIPPET_LENGTH) if diary_summary else None,
    }


def build_recent_daily_context(uid: str, days: int = 3) -> str:
    parts: List[str] = []
    labels = ["hari ini", "kemarin", "2 hari lalu"]

    for index, date_value in enumerate(_recent_date_ids(days)):
        metric_snapshot = _daily_metrics_collection(uid).document(date_value).get()
        metric_data = metric_snapshot.to_dict() or {}
        mood = _metric_mood(metric_data)
        sleep_payload = _metric_sleep_payload(metric_data)
        diary_summary = _diary_summary_for_date(uid, date_value)

        day_parts: List[str] = []
        day_parts.append(f"mood {mood}" if mood else "mood belum dicatat")
        if sleep_payload:
            quality = sleep_payload.get("sleep_quality") or "N/A"
            hours = sleep_payload.get("total_sleep_hours", 0)
            day_parts.append(f"tidur {hours:g} jam ({quality})")
        else:
            day_parts.append("data tidur belum dicatat")
        if diary_summary:
            day_parts.append(f"diary: \"{_snippet(diary_summary, CONTEXT_SNIPPET_LENGTH)}\"")

        label = labels[index] if index < len(labels) else f"{index} hari lalu"
        parts.append(f"{label} ({date_value}): " + "; ".join(day_parts))

    return " | ".join(parts)
