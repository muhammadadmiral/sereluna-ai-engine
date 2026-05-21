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


class _FirestoreEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, "item") and callable(obj.item):
            return obj.item()
        if hasattr(obj, "tolist") and callable(obj.tolist):
            return obj.tolist()
        try:
            if type(obj).__module__ == "numpy":
                return float(obj)
        except Exception:
            pass
        try:
            return str(obj)
        except Exception:
            return f"unserializable:{type(obj).__name__}"


def _is_firestore_sentinel(value: Any) -> bool:
    try:
        # Check if it's a known Firestore sentinel/transform object
        module = type(value).__module__
        name = type(value).__name__
        if "google.cloud.firestore" in module:
            if any(s in name for s in ("Sentinel", "Transform", "ServerTimestamp")):
                return True
    except Exception:
        pass
    return False


def _clean_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            (str(k).replace(".", "_").replace("/", "_") or "empty_key"): _clean_keys(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_clean_keys(v) for v in value]
    return value


def serialize_firestore_value(value: Any) -> Any:
    if value is None:
        return None

    # 1. If it's a sentinel, return as-is
    if _is_firestore_sentinel(value):
        return value

    # 2. Use JSON roundtrip to force basic Python types for everything else
    # This strips numpy types, sets, tuples, and other non-standard objects
    try:
        # For floats, handle NaN/Inf by converting to a string first or handling in encoder
        # Actually, json.dumps handles them if allow_nan=False, but it raises an error.
        # We'll use a safer approach:
        json_str = json.dumps(value, cls=_FirestoreEncoder, allow_nan=False)
        cleaned_value = json.loads(json_str)
    except (ValueError, TypeError):
        # Fallback if there are NaNs or other issues: use a more lenient dumps then clean
        json_str = json.dumps(value, cls=_FirestoreEncoder, allow_nan=True)
        cleaned_value = json.loads(json_str)
        # Manually fix NaNs/Infs in the cleaned value
        def _fix_numbers(v):
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return 0.0
            elif isinstance(v, list):
                return [_fix_numbers(i) for i in v]
            elif isinstance(v, dict):
                return {k: _fix_numbers(i) for k, i in v.items()}
            return v
        cleaned_value = _fix_numbers(cleaned_value)

    # 3. Final key sanitization (Firestore forbids . and / in keys)
    return _clean_keys(cleaned_value)


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
