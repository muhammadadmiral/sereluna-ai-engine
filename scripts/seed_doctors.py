import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.firebase_service import get_firestore_client, server_timestamp


REQUIRED_FIELDS = {"name", "specialty", "whatsapp_number"}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "doctor"


def _load_doctors(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    doctors = payload.get("doctors") if isinstance(payload, dict) else payload
    if not isinstance(doctors, list):
        raise ValueError("JSON must be an array or an object with a 'doctors' array.")

    cleaned = []
    for index, item in enumerate(doctors, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Doctor item #{index} must be an object.")

        missing = [field for field in REQUIRED_FIELDS if not str(item.get(field) or "").strip()]
        if missing:
            raise ValueError(f"Doctor item #{index} is missing required fields: {', '.join(missing)}")

        doctor = {
            "name": str(item["name"]).strip(),
            "specialty": str(item["specialty"]).strip(),
            "whatsapp_number": str(item["whatsapp_number"]).strip(),
            "image_url": str(item.get("image_url") or "").strip(),
            "updatedAt": server_timestamp(),
        }
        if "active" in item:
            doctor["active"] = bool(item["active"])

        doc_id = str(item.get("id") or _slug(doctor["name"])).strip()
        cleaned.append({"id": doc_id, "data": doctor})

    return cleaned


def seed_doctors(path: Path) -> None:
    doctors = _load_doctors(path)
    db = get_firestore_client()
    batch = db.batch()

    for doctor in doctors:
        ref = db.collection("doctors").document(doctor["id"])
        batch.set(ref, doctor["data"], merge=True)

    batch.commit()
    print(f"Uploaded {len(doctors)} doctor document(s) to Firestore collection 'doctors'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk upload Sereluna doctor data to Firestore.")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/doctors.local.json",
        help="Path to local JSON file. Default: data/doctors.local.json",
    )
    args = parser.parse_args()

    path = (ROOT / args.path).resolve() if not Path(args.path).is_absolute() else Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Doctor JSON file not found: {path}")

    seed_doctors(path)


if __name__ == "__main__":
    main()
