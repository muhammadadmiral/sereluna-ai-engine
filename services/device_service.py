from services.firebase_service import server_timestamp, user_document


def save_device_token(uid: str, token: str) -> bool:
    user_ref = user_document(uid)
    snapshot = user_ref.get()
    existing = snapshot.to_dict() or {}
    is_new_token = bool(token) and existing.get("fcmToken") != token
    user_ref.set(
        {
            "fcmToken": token,
            "fcmTokenUpdatedAt": server_timestamp(),
            "updatedAt": server_timestamp(),
        },
        merge=True,
    )
    return is_new_token
