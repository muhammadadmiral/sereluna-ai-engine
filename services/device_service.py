from services.firebase_service import server_timestamp, user_document


def save_device_token(uid: str, token: str) -> None:
    user_document(uid).set(
        {
            "fcmToken": token,
            "fcmTokenUpdatedAt": server_timestamp(),
            "updatedAt": server_timestamp(),
        },
        merge=True,
    )
