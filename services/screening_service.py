import csv
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from firebase_admin import firestore

from services.daily_dashboard_service import APP_TIMEZONE
from services.firebase_service import user_document
from services.notification_service import create_notification


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASS21_QUESTIONS_PATH = PROJECT_ROOT / "data" / "screening" / "dass21_questions.csv"
DASS21_RECOMMENDED_INTERVAL_DAYS = 7


@lru_cache(maxsize=1)
def get_dass21_questions() -> List[Dict[str, Any]]:
    with DASS21_QUESTIONS_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [
        {
            "id": int(row["id"]),
            "category": row["category"],
            "text": row["text"],
            "answer_min": 0,
            "answer_max": 3,
        }
        for row in rows
    ]


def get_dass21_questionnaire() -> Dict[str, Any]:
    return {
        "instrument": "DASS-21",
        "version": "bahasa_indonesia_csv_v1.0",
        "source_file": "data/screening/dass21_questions.csv",
        "recommended_interval_days": DASS21_RECOMMENDED_INTERVAL_DAYS,
        "disclaimer": "DASS-21 adalah alat screening, bukan diagnosis medis.",
        "instructions": "Pilih jawaban 0-3 sesuai kondisi yang paling menggambarkan satu minggu terakhir.",
        "answer_options": [
            {"value": 0, "label": "Tidak pernah / tidak sesuai dengan saya"},
            {"value": 1, "label": "Kadang-kadang / sesuai sampai tingkat tertentu"},
            {"value": 2, "label": "Sering / cukup sesuai dengan saya"},
            {"value": 3, "label": "Hampir selalu / sangat sesuai dengan saya"},
        ],
        "questions": get_dass21_questions(),
    }


def _parse_date(value: Optional[str]):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_screening_status(uid: str) -> Dict[str, Any]:
    latest: Optional[Dict[str, Any]] = None
    for snapshot in user_document(uid).collection("screenings").order_by(
        "date", direction=firestore.Query.DESCENDING
    ).limit(1).stream():
        data = snapshot.to_dict() or {}
        latest = {
            "date": str(data.get("date") or snapshot.id),
            "severity": data.get("severity") or {},
            "scores": data.get("scores") or {},
            "summary": data.get("summary") or "",
        }
        break

    server_time = _utc_now_iso()
    today = datetime.now(ZoneInfo(APP_TIMEZONE)).date()
    last_date = _parse_date(latest["date"]) if latest else None
    next_date = last_date + timedelta(days=DASS21_RECOMMENDED_INTERVAL_DAYS) if last_date else None
    is_due = next_date is None or today >= next_date
    next_in_days = max(0, (next_date - today).days) if next_date else 0
    if is_due and latest:
        create_notification(
            uid=uid,
            title="Skrining tersedia lagi",
            body="DASS-21 bisa diisi lagi untuk memperbarui baseline wellbeing mingguanmu.",
            notification_type="screening",
            priority="medium",
            category_label="Skrining",
            action_link="/screening/dass21",
            notification_key=f"screening_due:{next_date.isoformat() if next_date else 'initial'}",
        )

    return {
        "instrument": "DASS-21",
        "recommended_interval_days": DASS21_RECOMMENDED_INTERVAL_DAYS,
        "is_due": is_due,
        "latest": latest,
        "next_recommended_date": next_date.isoformat() if next_date else None,
        "next_recommended_in_days": next_in_days,
        "server_time": server_time,
        "updated_at": server_time,
        "disclaimer": "DASS-21 adalah alat screening, bukan diagnosis medis.",
    }
