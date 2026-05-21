import calendar
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from firebase_admin import firestore
from google.cloud.firestore_v1 import FieldFilter

from services.firebase_service import serialize_firestore_value, server_timestamp, user_document
from services.nlp_service import calculate_sentiment_score, classify_risk
from services.summary_service import clean_diary_summary


APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Jakarta")
MOOD_VALUES = {"happy", "neutral", "sad", "anxious", "angry"}
MOOD_SCORE = {"happy": 95, "neutral": 72, "anxious": 46, "sad": 38, "angry": 42}
DIARY_SNIPPET_LENGTH = 50
CONTEXT_SNIPPET_LENGTH = 90
WELLBEING_WEIGHTS = {"mood": 0.40, "sleep": 0.35, "diary": 0.25}
DAILY_WELLBEING_MODEL_VERSION = "daily_wellbeing_v1.0"


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
    diary_ref = _diaries_collection(uid).document(date_value)
    summary = _diary_summary_from_snapshot(diary_ref.get())
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
        date_value = str(data.get("date") or snapshot.id)
        return {
            "latest_date": date_value,
            "stress": str(severity.get("stress") or "unknown"),
            "anxiety": str(severity.get("anxiety") or "unknown"),
            "depression": str(severity.get("depression") or "unknown"),
            "scores": {
                "stress": int(scores.get("stress") or 0),
                "anxiety": int(scores.get("anxiety") or 0),
                "depression": int(scores.get("depression") or 0),
            },
            "label": "Baseline DASS-21 terakhir",
            "disclaimer": "Bukan diagnosis medis.",
        }
    return None


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


def _recent_date_ids(days: int = 3) -> List[str]:
    today = datetime.now(ZoneInfo(APP_TIMEZONE)).date()
    return [(today - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(max(1, days))]


def _clamp_score(value: float) -> int:
    return int(round(max(0, min(value, 100))))


def _sleep_score(sleep: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sleep:
        return None

    hours = float(sleep.get("total_sleep_hours") or 0)
    if 7 <= hours <= 9:
        score = 95
        reason = "durasi tidur berada di rentang ideal 7-9 jam"
    elif 6 <= hours < 7 or 9 < hours <= 10:
        score = 78
        reason = "durasi tidur sedikit di luar rentang ideal"
    elif 5 <= hours < 6 or 10 < hours <= 11:
        score = 58
        reason = "durasi tidur cukup jauh dari rentang ideal"
    else:
        score = 35
        reason = "durasi tidur sangat pendek atau terlalu panjang"

    quality = str(sleep.get("sleep_quality") or "").strip().lower()
    if quality in {"excellent", "very good", "good", "baik"}:
        score += 5
    elif quality in {"poor", "bad", "buruk", "very bad"}:
        score -= 12

    return {
        "name": "sleep",
        "score": _clamp_score(score),
        "weight": WELLBEING_WEIGHTS["sleep"],
        "reason": reason,
    }


def _mood_score(mood: Optional[str]) -> Optional[Dict[str, Any]]:
    if not mood:
        return None
    score = MOOD_SCORE.get(mood)
    if score is None:
        return None
    return {
        "name": "mood",
        "score": score,
        "weight": WELLBEING_WEIGHTS["mood"],
        "reason": f"mood harian dicatat sebagai {mood}",
    }


def _diary_score(diary_summary: str) -> Optional[Dict[str, Any]]:
    if not diary_summary:
        return None

    sentiment_score = calculate_sentiment_score(diary_summary)
    risk = classify_risk(diary_summary)
    sentiment_component = {1: 20, 2: 42, 3: 68, 4: 84, 5: 95}.get(sentiment_score, 68)
    risk_penalty = {"low": 0, "medium": 25, "high": 55}.get(risk["level"], 0)
    score = _clamp_score(sentiment_component - risk_penalty)

    return {
        "name": "diary",
        "score": score,
        "weight": WELLBEING_WEIGHTS["diary"],
        "reason": f"sentimen diary {sentiment_score}/5 dengan risiko {risk['level']}",
    }


def _level_for_score(score: Optional[int]) -> str:
    if score is None:
        return "no_data"
    if score >= 80:
        return "stable"
    if score >= 60:
        return "watch"
    if score >= 40:
        return "attention"
    return "high_attention"


def _recommendation_for(level: str, components: List[Dict[str, Any]]) -> Optional[str]:
    if level == "no_data":
        return "Lengkapi mood, tidur, atau diary agar Sereluna bisa membaca pola harian."
    if level == "stable":
        return "Pertahankan pola yang sudah baik dan tetap catat mood, tidur, serta diary secara konsisten."
    lowest = min(components, key=lambda item: item["score"], default=None)
    if not lowest:
        return None
    if lowest["name"] == "sleep":
        return "Prioritaskan rutinitas tidur yang lebih konsisten malam ini."
    if lowest["name"] == "mood":
        return "Coba catat pemicu mood hari ini dan pilih satu aktivitas pemulihan ringan."
    if lowest["name"] == "diary":
        return "Luangkan waktu untuk menulis lanjutan atau gunakan chat Sereluna jika beban terasa berat."
    return None


def _risk_level_for_components(components: List[Dict[str, Any]]) -> str:
    diary = next((component for component in components if component["name"] == "diary"), None)
    if diary and diary["score"] <= 30:
        return "medium"
    if any(component["score"] <= 35 for component in components):
        return "medium"
    return "low"


def _human_summary_for(mood: Optional[str], sleep: Optional[Dict[str, Any]], insight: Dict[str, Any]) -> str:
    if insight["score"] is None:
        return "Belum ada cukup data harian untuk membaca pola wellbeing."

    parts: List[str] = []
    if mood:
        mood_text = {
            "happy": "Mood terlihat positif",
            "neutral": "Mood cenderung netral",
            "anxious": "Mood cenderung tegang",
            "sad": "Mood cenderung turun",
            "angry": "Mood menunjukkan emosi intens",
        }.get(mood, f"Mood tercatat {mood}")
        parts.append(mood_text)

    if sleep:
        hours = float(sleep.get("total_sleep_hours") or 0)
        if hours < 6:
            parts.append("tidur kurang mendukung pemulihan")
        elif 7 <= hours <= 9:
            parts.append("tidur cukup mendukung pemulihan")
        else:
            parts.append("pola tidur perlu tetap dipantau")

    if not parts:
        return "Ada sinyal harian yang bisa dipantau dari diary atau data wellbeing."
    return ", ".join(parts).rstrip(".") + "."


def _signals_for(level: str, components: List[Dict[str, Any]]) -> List[str]:
    if not components:
        return ["Belum ada data harian yang cukup untuk dianalisis."]

    signals = [f"{item['name']}={item['score']}/100" for item in components]
    if level in {"attention", "high_attention"}:
        signals.append("Perlu perhatian karena ada komponen harian yang rendah.")
    elif level == "watch":
        signals.append("Kondisi cukup, tetapi masih ada sinyal yang perlu dipantau.")
    else:
        signals.append("Kondisi harian relatif stabil berdasarkan data yang tersedia.")
    return signals


def build_daily_wellbeing_insight(
    mood: Optional[str],
    sleep: Optional[Dict[str, Any]],
    diary_summary: str,
) -> Dict[str, Any]:
    components = [
        component
        for component in (
            _mood_score(mood),
            _sleep_score(sleep),
            _diary_score(diary_summary),
        )
        if component is not None
    ]

    if not components:
        score = None
    else:
        available_weight = sum(component["weight"] for component in components)
        weighted_score = sum(component["score"] * component["weight"] for component in components)
        score = _clamp_score(weighted_score / available_weight)

    level = _level_for_score(score)
    return {
        "score": score,
        "level": level,
        "signals": _signals_for(level, components),
        "recommendation": _recommendation_for(level, components),
        "risk_level": _risk_level_for_components(components),
        "model_version": DAILY_WELLBEING_MODEL_VERSION,
        "components": components,
        "algorithm": {
            "name": "Sereluna Daily Wellbeing Index",
            "version": "1.0",
            "method": "weighted rule-based scoring with available-component weight normalization",
            "weights": WELLBEING_WEIGHTS,
            "levels": {
                "stable": "80-100",
                "watch": "60-79",
                "attention": "40-59",
                "high_attention": "0-39",
                "no_data": "no component available",
            },
        },
    }


def _indicator_for_level(level: str) -> str:
    return {
        "stable": "green",
        "watch": "yellow",
        "attention": "orange",
        "high_attention": "red",
    }.get(level, "empty")


def _summary_item(
    date_value: str,
    metric_data: Optional[Dict[str, Any]] = None,
    diary_summary: str = "",
    has_diary: bool = False,
) -> Dict[str, Any]:
    metric_data = metric_data or {}
    mood = _metric_mood(metric_data)
    sleep_payload = _metric_sleep_payload(metric_data)
    insight = build_daily_wellbeing_insight(mood, sleep_payload, diary_summary)
    return {
        "date": date_value,
        "has_sleep_data": sleep_payload is not None,
        "mood": mood,
        "has_diary": has_diary,
        "wellbeing_score": insight["score"],
        "wellbeing_level": insight["level"],
        "indicator": _indicator_for_level(insight["level"]),
        "summary": _human_summary_for(mood, sleep_payload, insight),
        "recommendation": insight["recommendation"],
        "risk_level": insight["risk_level"],
        "model_version": insight["model_version"],
    }


def save_daily_mood(uid: str, date: str, mood: str) -> None:
    _daily_metrics_collection(uid).document(date).set(
        serialize_firestore_value(
            {
                "date": date,
                "mood": mood,
                "updatedAt": server_timestamp(),
            }
        ),
        merge=True,
    )


def list_calendar_summary(uid: str, year: int, month: int) -> List[Dict[str, Any]]:
    start_date, end_date = _month_bounds(year, month)
    items_by_date: Dict[str, Dict[str, Any]] = {}
    metrics_by_date: Dict[str, Dict[str, Any]] = {}

    metrics_ref = _daily_metrics_collection(uid)
    metrics_query = metrics_ref.where(filter=FieldFilter("date", ">=", start_date)).where(
        filter=FieldFilter("date", "<=", end_date)
    )
    for snapshot in metrics_query.stream():
        data = snapshot.to_dict() or {}
        date_value = data.get("date") or snapshot.id
        if not start_date <= date_value <= end_date:
            continue

        metrics_by_date[date_value] = data
        items_by_date[date_value] = _summary_item(date_value, metric_data=data)

    diaries_ref = _diaries_collection(uid)
    diaries_query = diaries_ref.where(filter=FieldFilter("date", ">=", start_date)).where(
        filter=FieldFilter("date", "<=", end_date)
    )
    for snapshot in diaries_query.stream():
        data = snapshot.to_dict() or {}
        date_value = data.get("date") or snapshot.id
        if not start_date <= date_value <= end_date:
            continue

        summary = clean_diary_summary(data.get("chatSummary") or data.get("chat_summary") or data.get("summary") or "")
        items_by_date[date_value] = _summary_item(
            date_value,
            metric_data=metrics_by_date.get(date_value, {}),
            diary_summary=summary,
            has_diary=True,
        )

    screening_context = get_latest_screening_context(uid)
    items = [items_by_date[date_value] for date_value in sorted(items_by_date)]
    if screening_context:
        for item in items:
            item["screening_context"] = screening_context
    return items


def get_calendar_detail(uid: str, date: str) -> Dict[str, Any]:
    metric_snapshot = _daily_metrics_collection(uid).document(date).get()
    metric_data = metric_snapshot.to_dict() or {}
    diary_summary = _diary_summary_for_date(uid, date)

    mood = _metric_mood(metric_data)
    sleep_payload = _metric_sleep_payload(metric_data)
    insight = build_daily_wellbeing_insight(mood, sleep_payload, diary_summary)

    return {
        "date": date,
        "mood": mood,
        "sleep": sleep_payload,
        "diary_snippet": _snippet(diary_summary, DIARY_SNIPPET_LENGTH) if diary_summary else None,
        "summary": _human_summary_for(mood, sleep_payload, insight),
        "indicator": _indicator_for_level(insight["level"]),
        "screening_context": get_latest_screening_context(uid),
        "wellbeing": insight,
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
        insight = build_daily_wellbeing_insight(mood, sleep_payload, diary_summary)

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
        if insight["score"] is not None:
            day_parts.append(f"wellbeing {insight['score']}/100 ({insight['level']})")

        label = labels[index] if index < len(labels) else f"{index} hari lalu"
        parts.append(f"{label} ({date_value}): " + "; ".join(day_parts))

    return " | ".join(parts)
