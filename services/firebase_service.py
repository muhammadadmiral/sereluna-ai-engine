import json
import math
import os
from datetime import date, datetime
from typing import Any, Dict, Optional

import firebase_admin
from dotenv import load_dotenv
from fastapi import Header, HTTPException, status
from firebase_admin import auth, credentials, firestore

load_dotenv()


def _resolve_firebase_project_id(service_account_json: Optional[str] = None, service_account_path: Optional[str] = None) -> Optional[str]:
    project_id = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if project_id:
        return project_id.strip()

    if service_account_json:
        try:
            service_account_info = json.loads(service_account_json)
        except json.JSONDecodeError:
            return None
        project_id = service_account_info.get("project_id")
        if project_id:
            return str(project_id).strip()

    if service_account_path and os.path.exists(service_account_path):
        try:
            with open(service_account_path, "r", encoding="utf-8") as handle:
                service_account_info = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        project_id = service_account_info.get("project_id")
        if project_id:
            return str(project_id).strip()

    return None


def initialize_firebase() -> None:
    try:
        firebase_admin.get_app()
        return
    except ValueError:
        pass

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    project_id = _resolve_firebase_project_id(service_account_json, service_account_path)
    app_options = {"projectId": project_id} if project_id else None

    if service_account_json:
        try:
            service_account_info = json.loads(service_account_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred, app_options)
        return

    if service_account_path:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred, app_options)
        return

    firebase_admin.initialize_app(options=app_options)


def get_firestore_client():
    initialize_firebase()
    return firestore.client()


def server_timestamp():
    return firestore.SERVER_TIMESTAMP


def user_document(uid: str):
    return get_firestore_client().collection("users").document(uid)


def serialize_firestore_value(value: Any) -> Any:
    if value is None:
        return None

    # 1. Handle basic primitives
    if isinstance(value, str):
        return value
    if isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value

    # 2. Handle known serializable objects
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    # 3. Handle numpy types specifically
    # Check for item() method (numpy scalars)
    if hasattr(value, "item") and callable(value.item):
        try:
            return serialize_firestore_value(value.item())
        except Exception:
            pass

    # Check for tolist() method (numpy arrays)
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return [serialize_firestore_value(item) for item in value.tolist()]
        except Exception:
            pass

    # 4. Handle containers (recursive)
    if isinstance(value, dict):
        serialized_dict = {}
        for key, item in value.items():
            # Sanitize key: force string, handle empty, replace dots/slashes
            safe_key = str(key) if key is not None else "None"
            if not safe_key:
                safe_key = "empty_key"
            # Dots and slashes are forbidden in Firestore field names
            safe_key = safe_key.replace(".", "_").replace("/", "_")
            serialized_dict[safe_key] = serialize_firestore_value(item)
        return serialized_dict

    if isinstance(value, (list, tuple, set)):
        return [serialize_firestore_value(item) for item in value]

    # 5. Handle special Firestore types (sentinels like SERVER_TIMESTAMP)
    # These must NOT be modified
    type_name = type(value).__name__
    if type_name in ("Sentinel", "ServerTimestamp"):
        return value

    # 6. Fallback for other potential numpy types or objects
    try:
        if type(value).__module__ == "numpy":
            if hasattr(value, "tolist"):
                return serialize_firestore_value(value.tolist())
            return float(value)
    except Exception:
        pass

    # 7. Final resort: convert to string to avoid Firestore 'invalid nested entity' error
    try:
        return str(value)
    except Exception:
        return f"unserializable:{type(value).__name__}"


def verify_id_token(id_token: str) -> Dict[str, Any]:
    initialize_firebase()
    return auth.verify_id_token(id_token)


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be Bearer token",
        )

    try:
        decoded = verify_id_token(token.strip())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase ID token",
        ) from exc

    uid = decoded.get("uid") or decoded.get("user_id")
    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firebase token does not contain uid",
        )

    return {
        "uid": uid,
        "email": decoded.get("email", ""),
        "name": decoded.get("name", ""),
        "claims": decoded,
    }
