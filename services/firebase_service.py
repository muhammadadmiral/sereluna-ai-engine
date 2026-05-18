import json
import os
from typing import Any, Dict, Optional

import firebase_admin
from dotenv import load_dotenv
from fastapi import Header, HTTPException, status
from firebase_admin import auth, credentials, firestore

load_dotenv()


def initialize_firebase() -> None:
    try:
        firebase_admin.get_app()
        return
    except ValueError:
        pass

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")

    if service_account_json:
        try:
            service_account_info = json.loads(service_account_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        return

    if service_account_path:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
        return

    firebase_admin.initialize_app()


def get_firestore_client():
    initialize_firebase()
    return firestore.client()


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
