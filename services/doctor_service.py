from typing import Any, Dict, List

from services.firebase_service import get_firestore_client, serialize_firestore_value


def _doctor_from_snapshot(snapshot: Any) -> Dict[str, Any]:
    data = snapshot.to_dict() or {}
    return {
        "id": snapshot.id,
        "name": str(data.get("name") or ""),
        "specialty": str(data.get("specialty") or ""),
        "whatsapp_number": str(data.get("whatsapp_number") or ""),
        "image_url": str(data.get("image_url") or ""),
    }


def list_doctors() -> List[Dict[str, Any]]:
    doctors_ref = get_firestore_client().collection("doctors")
    doctors = []

    for snapshot in doctors_ref.order_by("name").stream():
        doctor = _doctor_from_snapshot(snapshot)
        if doctor["name"] and doctor["whatsapp_number"]:
            doctors.append(serialize_firestore_value(doctor))

    return doctors
