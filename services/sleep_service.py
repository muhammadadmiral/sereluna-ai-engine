from typing import Any, Dict, List

from firebase_admin import firestore

from services.firebase_service import serialize_firestore_value, server_timestamp, user_document


def save_daily_sleep_metric(
    uid: str,
    date: str,
    bedtime: str,
    wakeup: str,
    sleep_quality: str,
    total_sleep_hours: float,
) -> None:
    metric_ref = (
        user_document(uid)
        .collection("analytics")
        .document("dailyMetrics")
        .collection("dates")
        .document(date)
    )
    metric_ref.set(
        {
            "date": date,
            "bedtime": bedtime,
            "wakeup": wakeup,
            "sleepQuality": sleep_quality,
            "totalSleepHours": total_sleep_hours,
            "updatedAt": server_timestamp(),
        },
        merge=True,
    )


def list_daily_sleep_metrics(uid: str, limit: int = 14) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(limit, 60))
    metrics_ref = user_document(uid).collection("analytics").document("dailyMetrics").collection("dates")
    items: List[Dict[str, Any]] = []
    for snapshot in metrics_ref.order_by("date", direction=firestore.Query.DESCENDING).limit(safe_limit).stream():
        data = snapshot.to_dict() or {}
        items.append(
            {
                "date": data.get("date") or snapshot.id,
                "sleep_quality": data.get("sleepQuality") or data.get("sleep_quality") or "",
                "total_sleep_hours": float(data.get("totalSleepHours") or data.get("total_sleep_hours") or 0),
                "bedtime": serialize_firestore_value(data.get("bedtime") or data.get("bedtimeAt")),
                "wakeup": serialize_firestore_value(data.get("wakeup") or data.get("wakeupAt")),
                "updated_at": serialize_firestore_value(data.get("updatedAt") or data.get("updated_at")),
            }
        )
    return items
