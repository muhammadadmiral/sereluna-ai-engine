from collections import Counter
from datetime import datetime, timedelta
from statistics import pstdev
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from google.cloud.firestore_v1 import FieldFilter

from services.daily_dashboard_service import APP_TIMEZONE, MOOD_VALUES
from services.firebase_service import user_document


MOOD_ORDER = ["happy", "neutral", "sad", "anxious", "angry"]


def _daily_metrics_collection(uid: str):
    return user_document(uid).collection("analytics").document("dailyMetrics").collection("dates")


def _date_bounds(days: int) -> tuple[str, str]:
    safe_days = max(1, min(days, 90))
    today = datetime.now(ZoneInfo(APP_TIMEZONE)).date()
    start_date = today - timedelta(days=safe_days - 1)
    return start_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def _metric_rows(uid: str, days: int) -> List[Dict[str, Any]]:
    start_date, end_date = _date_bounds(days)
    query = _daily_metrics_collection(uid).where(filter=FieldFilter("date", ">=", start_date)).where(
        filter=FieldFilter("date", "<=", end_date)
    )
    rows: List[Dict[str, Any]] = []
    for snapshot in query.stream():
        data = snapshot.to_dict() or {}
        date_value = str(data.get("date") or snapshot.id)
        if start_date <= date_value <= end_date:
            rows.append({**data, "date": date_value})
    return rows


def _mood_insight(dominant_mood: Optional[str], total_entries: int, days: int) -> str:
    if total_entries == 0:
        return "Belum ada data mood yang cukup untuk ditampilkan."
    period = "minggu ini" if days <= 7 else "periode ini"
    labels = {
        "happy": "Mood senang paling sering tercatat.",
        "neutral": "Mood netral paling sering tercatat.",
        "anxious": "Mood cemas cukup sering tercatat.",
        "sad": "Mood sedih cukup sering tercatat.",
        "angry": "Mood marah cukup sering tercatat.",
    }
    return f"Pada {period}, {labels.get(dominant_mood, 'pola mood tersedia untuk dilihat.')}"


def get_mood_distribution(uid: str, days: int) -> Dict[str, Any]:
    safe_days = max(1, min(days, 90))
    counter: Counter[str] = Counter()
    for row in _metric_rows(uid, safe_days):
        mood = str(row.get("mood") or "").strip().lower()
        if mood in MOOD_VALUES:
            counter[mood] += 1

    dominant_mood = max(MOOD_ORDER, key=lambda mood: counter[mood]) if counter else None
    return {
        "period_days": safe_days,
        "data": [{"mood": mood, "count": counter[mood]} for mood in MOOD_ORDER if counter[mood] > 0],
        "dominant_mood": dominant_mood,
        "insight": _mood_insight(dominant_mood, sum(counter.values()), safe_days),
    }


def _sleep_hours(row: Dict[str, Any]) -> Optional[float]:
    raw_value = row.get("totalSleepHours", row.get("total_sleep_hours"))
    try:
        hours = float(raw_value)
    except (TypeError, ValueError):
        return None
    return round(hours, 2) if 0 <= hours <= 24 else None


def _sleep_insight(hours: List[float]) -> str:
    if not hours:
        return "Belum ada catatan tidur yang cukup untuk ditampilkan."
    average = sum(hours) / len(hours)
    deviation = pstdev(hours) if len(hours) > 1 else 0
    if deviation >= 1.25:
        return f"Durasi tidur tercatat rata-rata {average:.1f} jam dengan pola yang berubah-ubah."
    return f"Durasi tidur tercatat rata-rata {average:.1f} jam pada periode ini."


def get_sleep_trends(uid: str, days: int) -> Dict[str, Any]:
    safe_days = max(1, min(days, 90))
    items = [
        {"date": row["date"], "hours": hours}
        for row in _metric_rows(uid, safe_days)
        if (hours := _sleep_hours(row)) is not None
    ]
    items.sort(key=lambda item: item["date"])
    hour_values = [item["hours"] for item in items]
    return {
        "average_hours": round(sum(hour_values) / len(hour_values), 1) if hour_values else 0.0,
        "items": items,
        "insight": _sleep_insight(hour_values),
    }
