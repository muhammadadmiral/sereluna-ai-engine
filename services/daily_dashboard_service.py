import calendar
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from firebase_admin import firestore
from google.cloud.firestore_v1 import FieldFilter

from services.firebase_service import serialize_firestore_value, server_timestamp, user_document
from services.summary_service import clean_diary_summary


APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Jakarta")
MOOD_VALUES = {"happy", "neutral", "sad", "anxious", "angry"}
DIARY_SNIPPET_LENGTH = 50
CONTEXT_SNIPPET_LENGTH = 90


def _daily_metrics_collection(uid: str):
    return user_document(uid).collection("analytics").document("dailyMetrics").collection("dates")


def _diaries_collection(uid: str):
    return user_document(uid).collection("diaries")


def _screenings_collection(uid: str):
    return user_document(uid).collection("screenings")


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def _snippet(value: str, limit: int) -> str:
    normalized = _normalize_text(value)
    return normalized if len(normalized) <= limit else normalized[:limit].rstrip() + "..."


def _metric_sleep_payload(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sleep_quality = data.get("sleepQuality") or data.get("sleep_quality") or ""
    total_sleep_hours = data.get("totalSleepHours", data.get("total_sleep_hours"))
    bedtime = data.get("bedtime") or data.get("bedtimeAt")
    wakeup = data.get("wakeup") or data.get("wakeupAt")
    if total_sleep_hours is None and not any((sleep_quality, bedtime, wakeup)):
        return None

    try:
        sleep_hours = float(total_sleep_hours or 0)
    except (TypeError, ValueError):
        sleep_hours = 0

    return {
        "total_sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "bedtime": serialize_firestore_value(bedtime),
        "wakeup": serialize_firestore_value(wakeup),
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
    summary = _diary_summary_from_snapshot(_diaries_collection(uid).document(date_value).get())
    if summary:
        return summary

    for snapshot in _diaries_collection(uid).where(filter=FieldFilter("date", "==", date_value)).limit(1).stream():
        summary = _diary_summary_from_snapshot(snapshot)
        if summary:
            return summary
    return ""


def get_latest_screening_context(uid: str) -> Optional[Dict[str, Any]]:
    for snapshot in _screenings_collection(uid).order_by("date", direction=firestore.Query.DESCENDING).limit(1).stream():
        data = snapshot.to_dict() or {}
        severity = data.get("severity") or {}
        scores = data.get("scores") or {}
        return {
            "latest_date": str(data.get("date") or snapshot.id),
            "stress": str(severity.get("stress") or "unknown"),
            "anxiety": str(severity.get("anxiety") or "unknown"),
            "depression": str(severity.get("depression") or "unknown"),
            "scores": {
                "stress": int(scores.get("stress") or 0),
                "anxiety": int(scores.get("anxiety") or 0),
                "depression": int(scores.get("depression") or 0),
            },
            "label": "Hasil DASS-21 terakhir",
            "disclaimer": "Bukan diagnosis medis.",
        }
    return None


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


def _recent_date_ids(days: int = 3) -> List[str]:
    today = datetime.now(ZoneInfo(APP_TIMEZONE)).date()
    return [(today - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(max(1, days))]


def _daily_summary(mood: Optional[str], sleep: Optional[Dict[str, Any]], has_diary: bool) -> str:
    parts: List[str] = []
    if mood:
        label = {"happy": "senang", "neutral": "netral", "sad": "sedih", "anxious": "cemas", "angry": "marah"}[mood]
        parts.append(f"mood tercatat {label}")
    if sleep:
        parts.append(f"tidur tercatat {float(sleep['total_sleep_hours']):g} jam")
    if has_diary:
        parts.append("diary tersedia")
    return "; ".join(parts).capitalize() + "." if parts else "Belum ada catatan harian untuk tanggal ini."


def _summary_item(
    date_value: str,
    metric_data: Optional[Dict[str, Any]] = None,
    diary_summary: str = "",
    has_diary: bool = False,
) -> Dict[str, Any]:
    metric_data = metric_data or {}
    mood = _metric_mood(metric_data)
    sleep_payload = _metric_sleep_payload(metric_data)
    return {
        "date": date_value,
        "has_sleep_data": sleep_payload is not None,
        "mood": mood,
        "has_diary": has_diary,
        "indicator": "recorded" if mood or sleep_payload or has_diary else "empty",
        "summary": _daily_summary(mood, sleep_payload, has_diary),
    }


def save_daily_mood(uid: str, date: str, mood: str) -> None:
    _daily_metrics_collection(uid).document(date).set(
        serialize_firestore_value({"date": date, "mood": mood, "updatedAt": server_timestamp()}), merge=True
    )


def list_calendar_summary(uid: str, year: int, month: int) -> List[Dict[str, Any]]:
    start_date, end_date = _month_bounds(year, month)
    items_by_date: Dict[str, Dict[str, Any]] = {}
    metrics_by_date: Dict[str, Dict[str, Any]] = {}

    for snapshot in _daily_metrics_collection(uid).where(filter=FieldFilter("date", ">=", start_date)).where(
        filter=FieldFilter("date", "<=", end_date)
    ).stream():
        data = snapshot.to_dict() or {}
        date_value = str(data.get("date") or snapshot.id)
        if start_date <= date_value <= end_date:
            metrics_by_date[date_value] = data
            items_by_date[date_value] = _summary_item(date_value, metric_data=data)

    for snapshot in _diaries_collection(uid).where(filter=FieldFilter("date", ">=", start_date)).where(
        filter=FieldFilter("date", "<=", end_date)
    ).stream():
        data = snapshot.to_dict() or {}
        date_value = str(data.get("date") or snapshot.id)
        if start_date <= date_value <= end_date:
            summary = clean_diary_summary(data.get("chatSummary") or data.get("chat_summary") or data.get("summary") or "")
            items_by_date[date_value] = _summary_item(
                date_value, metric_data=metrics_by_date.get(date_value, {}), diary_summary=summary, has_diary=True
            )

    screening_context = get_latest_screening_context(uid)
    items = [items_by_date[date_value] for date_value in sorted(items_by_date)]
    for item in items:
        item["screening_context"] = screening_context
    return items


def get_calendar_detail(uid: str, date: str) -> Dict[str, Any]:
    metric_data = _daily_metrics_collection(uid).document(date).get().to_dict() or {}
    diary_summary = _diary_summary_for_date(uid, date)
    mood = _metric_mood(metric_data)
    sleep_payload = _metric_sleep_payload(metric_data)
    return {
        "date": date,
        "mood": mood,
        "sleep": sleep_payload,
        "diary_snippet": _snippet(diary_summary, DIARY_SNIPPET_LENGTH) if diary_summary else None,
        "summary": _daily_summary(mood, sleep_payload, bool(diary_summary)),
        "indicator": "recorded" if mood or sleep_payload or diary_summary else "empty",
        "screening_context": get_latest_screening_context(uid),
    }


def build_recent_daily_context(uid: str, days: int = 3) -> str:
    labels = ["hari ini", "kemarin", "2 hari lalu"]
    parts: List[str] = []
    for index, date_value in enumerate(_recent_date_ids(days)):
        metric_data = _daily_metrics_collection(uid).document(date_value).get().to_dict() or {}
        mood = _metric_mood(metric_data)
        sleep_payload = _metric_sleep_payload(metric_data)
        diary_summary = _diary_summary_for_date(uid, date_value)
        day_parts = [f"mood {mood}" if mood else "mood belum dicatat"]
        if sleep_payload:
            quality = sleep_payload.get("sleep_quality") or "N/A"
            day_parts.append(f"tidur {sleep_payload['total_sleep_hours']:g} jam ({quality})")
        if diary_summary:
            day_parts.append(f'diary: "{_snippet(diary_summary, CONTEXT_SNIPPET_LENGTH)}"')
        label = labels[index] if index < len(labels) else f"{index} hari lalu"
        parts.append(f"{label} ({date_value}): " + "; ".join(day_parts))
    return " | ".join(parts)
