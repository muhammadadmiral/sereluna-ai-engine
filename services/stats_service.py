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


def _period_label(days: int) -> str:
    return "minggu ini" if days <= 7 else "periode ini"


def _mood_insight(dominant_mood: Optional[str], total_entries: int, days: int) -> str:
    if total_entries == 0:
        return "Belum ada data mood yang cukup buat dibaca. Coba catat mood beberapa hari dulu biar grafiknya mulai kelihatan."

    label = _period_label(days)
    if dominant_mood == "happy":
        return f"{label.capitalize()} mood happy paling sering muncul. Polanya lagi cukup positif, jadi pertahankan hal-hal kecil yang bikin kamu stabil."
    if dominant_mood == "neutral":
        return f"{label.capitalize()} mood kamu banyak berada di area netral. Ini bisa berarti kondisi cukup stabil, tapi tetap perhatikan momen yang bikin energi naik atau turun."
    if dominant_mood == "anxious":
        return f"{label.capitalize()} rasa anxious cukup sering muncul. Coba cek pemicu yang berulang, lalu mulai dari langkah kecil seperti napas pelan, journaling singkat, atau istirahat sebentar."
    if dominant_mood == "sad":
        return f"{label.capitalize()} mood sad cukup dominan. Kalau pola ini terasa berat, jangan dipendam sendirian; kamu bisa lanjut cerita di chat atau cari dukungan orang terdekat."
    if dominant_mood == "angry":
        return f"{label.capitalize()} mood angry cukup sering tercatat. Coba beri jeda sebelum merespons hal yang memicu emosi, lalu catat situasi apa yang paling sering muncul."
    return f"Ada {total_entries} catatan mood dalam {days} hari terakhir. Lanjutkan pencatatan agar Sereluna bisa membaca polanya lebih akurat."


def get_mood_distribution(uid: str, days: int) -> Dict[str, Any]:
    safe_days = max(1, min(days, 90))
    counter: Counter[str] = Counter()
    for row in _metric_rows(uid, safe_days):
        mood = str(row.get("mood") or "").strip().lower()
        if mood in MOOD_VALUES:
            counter[mood] += 1

    data = [{"mood": mood, "count": counter[mood]} for mood in MOOD_ORDER if counter[mood] > 0]
    dominant_mood = max(MOOD_ORDER, key=lambda mood: counter[mood]) if counter else None
    total_entries = sum(counter.values())

    return {
        "period_days": safe_days,
        "data": data,
        "dominant_mood": dominant_mood,
        "insight": _mood_insight(dominant_mood, total_entries, safe_days),
    }


def _sleep_hours(row: Dict[str, Any]) -> Optional[float]:
    raw_value = row.get("totalSleepHours", row.get("total_sleep_hours"))
    if raw_value is None:
        return None
    try:
        hours = float(raw_value)
    except (TypeError, ValueError):
        return None
    if hours < 0 or hours > 24:
        return None
    return round(hours, 2)


def _sleep_insight(hours: List[float], days: int) -> str:
    if not hours:
        return "Belum ada data tidur yang cukup buat ditampilkan. Isi catatan tidur harian dulu supaya grafiknya mulai kebaca."

    average = sum(hours) / len(hours)
    spread = max(hours) - min(hours) if len(hours) > 1 else 0
    deviation = pstdev(hours) if len(hours) > 1 else 0
    unstable = spread >= 2.0 or deviation >= 1.25
    label = _period_label(days)

    if average < 6:
        return f"Rata-rata tidur kamu {average:.1f} jam di {label}, masih cenderung kurang. Coba majukan waktu tidur sedikit demi sedikit, bukan langsung berubah ekstrem."
    if average > 9:
        return f"Rata-rata tidur kamu {average:.1f} jam di {label}, agak panjang dari rentang ideal. Kalau tetap lelah setelah tidur lama, pola aktivitas dan kualitas tidurnya perlu diperhatikan."
    if unstable:
        return f"Durasi tidur kamu cukup oke secara rata-rata, tapi naik-turunnya masih terasa. Coba bikin jam tidur dan bangun yang lebih konsisten beberapa hari ke depan."
    return f"Pola tidur kamu relatif stabil di {label}. Rata-ratanya sudah dekat rentang ideal, jadi pertahankan rutinitas yang sekarang berjalan."


def get_sleep_trends(uid: str, days: int) -> Dict[str, Any]:
    safe_days = max(1, min(days, 90))
    items: List[Dict[str, Any]] = []
    for row in _metric_rows(uid, safe_days):
        hours = _sleep_hours(row)
        if hours is None:
            continue
        items.append({"date": row["date"], "hours": hours})

    items.sort(key=lambda item: item["date"])
    hour_values = [item["hours"] for item in items]
    average_hours = round(sum(hour_values) / len(hour_values), 1) if hour_values else 0.0

    return {
        "average_hours": average_hours,
        "items": items,
        "insight": _sleep_insight(hour_values, safe_days),
    }
