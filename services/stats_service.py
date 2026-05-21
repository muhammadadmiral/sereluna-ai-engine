from collections import Counter
from datetime import datetime, timedelta
from statistics import pstdev
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from firebase_admin import firestore
from google.cloud.firestore_v1 import FieldFilter

from services.daily_dashboard_service import (
    APP_TIMEZONE,
    MOOD_VALUES,
    build_daily_wellbeing_insight,
    get_latest_screening_context,
)
from services.firebase_service import user_document
from services.summary_service import clean_diary_summary


MOOD_ORDER = ["happy", "neutral", "sad", "anxious", "angry"]


def _daily_metrics_collection(uid: str):
    return user_document(uid).collection("analytics").document("dailyMetrics").collection("dates")


def _diaries_collection(uid: str):
    return user_document(uid).collection("diaries")


def _date_bounds(days: int) -> tuple[str, str]:
    safe_days = max(1, min(days, 90))
    today = datetime.now(ZoneInfo(APP_TIMEZONE)).date()
    start_date = today - timedelta(days=safe_days - 1)
    return start_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def days_from_range(range_value: str) -> int:
    normalized = (range_value or "30d").strip().lower()
    return {"7d": 7, "30d": 30, "90d": 90}.get(normalized, 30)


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


def _diary_rows(uid: str, days: int) -> Dict[str, str]:
    start_date, end_date = _date_bounds(days)
    query = _diaries_collection(uid).where(filter=FieldFilter("date", ">=", start_date)).where(
        filter=FieldFilter("date", "<=", end_date)
    )
    rows: Dict[str, str] = {}
    for snapshot in query.stream():
        data = snapshot.to_dict() or {}
        date_value = str(data.get("date") or snapshot.id)
        if not start_date <= date_value <= end_date:
            continue

        summary = clean_diary_summary(data.get("chatSummary") or data.get("chat_summary") or data.get("summary") or "")
        if not summary:
            for session_snapshot in snapshot.reference.collection("sessions").order_by(
                "updatedAt", direction=firestore.Query.DESCENDING
            ).limit(1).stream():
                session_data = session_snapshot.to_dict() or {}
                summary = clean_diary_summary(session_data.get("summary") or "")
                if summary:
                    break
        if summary:
            rows[date_value] = summary
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


def _sleep_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hours = _sleep_hours(row)
    sleep_quality = row.get("sleepQuality") or row.get("sleep_quality") or ""
    if hours is None and not sleep_quality:
        return None
    return {
        "total_sleep_hours": hours or 0,
        "sleep_quality": sleep_quality,
    }


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


def _overall_mood_label(average_score: Optional[float], dominant_mood: Optional[str]) -> str:
    if average_score is None:
        return "belum cukup data"
    if average_score >= 80:
        return "cenderung stabil positif"
    if average_score >= 60:
        return "cenderung stabil"
    if dominant_mood == "anxious":
        return "cenderung tegang"
    if dominant_mood == "sad":
        return "cenderung menurun"
    return "perlu perhatian"


def _wellbeing_insights(
    average_score: Optional[float],
    mood_counter: Counter[str],
    sleep_hours: List[float],
    diary_count: int,
) -> List[str]:
    insights: List[str] = []
    if average_score is None:
        return ["Belum ada data wellbeing yang cukup untuk dianalisis."]

    if sleep_hours and sum(1 for hours in sleep_hours if hours < 6) >= max(1, len(sleep_hours) // 3):
        insights.append("Mood dan wellbeing perlu dipantau saat durasi tidur berada di bawah 6 jam.")
    if mood_counter.get("anxious", 0) > mood_counter.get("happy", 0):
        insights.append("Mood anxious muncul cukup sering dalam periode ini.")
    if diary_count:
        insights.append("Diary ikut dipakai sebagai sinyal harian melalui sentiment dan risk scoring.")
    if not insights:
        insights.append("Pola wellbeing periode ini relatif stabil berdasarkan mood, tidur, dan diary yang tersedia.")
    return insights


def get_wellbeing_statistics(uid: str, range_value: str) -> Dict[str, Any]:
    days = days_from_range(range_value)
    normalized_range = f"{days}d"
    metric_rows = _metric_rows(uid, days)
    diary_by_date = _diary_rows(uid, days)
    metrics_by_date = {row["date"]: row for row in metric_rows}
    date_values = sorted(set(metrics_by_date) | set(diary_by_date))

    mood_counter: Counter[str] = Counter()
    wellbeing_scores: List[int] = []
    sleep_values: List[float] = []
    daily_items: List[Dict[str, Any]] = []

    for date_value in date_values:
        metric = metrics_by_date.get(date_value, {})
        mood = str(metric.get("mood") or "").strip().lower()
        if mood in MOOD_VALUES:
            mood_counter[mood] += 1
        sleep = _sleep_payload(metric)
        if sleep:
            hours = _sleep_hours(metric)
            if hours is not None:
                sleep_values.append(hours)

        insight = build_daily_wellbeing_insight(
            mood=mood if mood in MOOD_VALUES else None,
            sleep=sleep,
            diary_summary=diary_by_date.get(date_value, ""),
        )
        if insight["score"] is not None:
            wellbeing_scores.append(insight["score"])
        daily_items.append(
            {
                "date": date_value,
                "mood": mood if mood in MOOD_VALUES else None,
                "wellbeing_score": insight["score"],
                "wellbeing_level": insight["level"],
                "risk_level": insight["risk_level"],
            }
        )

    average_score = round(sum(wellbeing_scores) / len(wellbeing_scores), 1) if wellbeing_scores else None
    dominant_mood = max(MOOD_ORDER, key=lambda item: mood_counter[item]) if mood_counter else None

    return {
        "range": normalized_range,
        "period_days": days,
        "overall_mood": _overall_mood_label(average_score, dominant_mood),
        "average_wellbeing_score": average_score,
        "mood_distribution": {mood: mood_counter[mood] for mood in MOOD_ORDER},
        "dominant_mood": dominant_mood,
        "screening_context": get_latest_screening_context(uid),
        "insights": _wellbeing_insights(average_score, mood_counter, sleep_values, len(diary_by_date)),
        "daily_items": daily_items,
        "model_version": "wellbeing_statistics_v1.0",
        "disclaimer": "Insight ini bukan diagnosis medis.",
    }
