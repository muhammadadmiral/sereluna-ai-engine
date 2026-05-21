import json
import os
import urllib.error
import urllib.request
from typing import Optional

import firebase_admin
from firebase_admin import auth

from firebase_admin import credentials


def _resolve_project_id(service_account_info: Optional[dict] = None) -> Optional[str]:
    project_id = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if project_id:
        return project_id.strip()
    if service_account_info:
        value = service_account_info.get("project_id")
        if value:
            return str(value).strip()
    return None


def _get_password_reset_app() -> firebase_admin.App:
    app_name = "password-reset"
    try:
        return firebase_admin.get_app(app_name)
    except ValueError:
        pass

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    service_account_info = None

    if service_account_json:
        try:
            service_account_info = json.loads(service_account_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc
    elif service_account_path:
        with open(service_account_path, "r", encoding="utf-8") as handle:
            service_account_info = json.load(handle)

    if service_account_info:
        project_id = _resolve_project_id(service_account_info)
        app_options = {"projectId": project_id} if project_id else None
        cred = credentials.Certificate(service_account_info)
        return firebase_admin.initialize_app(cred, app_options, name=app_name)

    project_id = _resolve_project_id()
    app_options = {"projectId": project_id} if project_id else None
    return firebase_admin.initialize_app(options=app_options, name=app_name)


def generate_password_reset_link(email: str, continue_url: Optional[str] = None) -> str:
    app = _get_password_reset_app()

    action_code_settings = None
    if continue_url:
        settings_kwargs = {
            "url": continue_url,
            "handle_code_in_app": True,
        }
        android_package_name = os.getenv("ANDROID_PACKAGE_NAME") or ""
        if android_package_name:
            settings_kwargs["android_package_name"] = android_package_name
            settings_kwargs["android_install_app"] = True
            android_minimum_version = os.getenv("ANDROID_MINIMUM_VERSION") or ""
            if android_minimum_version:
                settings_kwargs["android_minimum_version"] = android_minimum_version
        link_domain = os.getenv("FIREBASE_LINK_DOMAIN") or ""
        if link_domain:
            settings_kwargs["link_domain"] = link_domain

        action_code_settings = auth.ActionCodeSettings(**settings_kwargs)

    return auth.generate_password_reset_link(
        email=email.strip(),
        action_code_settings=action_code_settings,
        app=app,
    )


class PasswordChangeError(Exception):
    pass


class InvalidOldPasswordError(PasswordChangeError):
    pass


class PasswordAuthConfigError(PasswordChangeError):
    pass


def _firebase_web_api_key() -> str:
    return (os.getenv("FIREBASE_WEB_API_KEY") or os.getenv("FIREBASE_API_KEY") or "").strip()


def _verify_email_password(email: str, password: str) -> None:
    api_key = _firebase_web_api_key()
    if not api_key:
        raise PasswordAuthConfigError("FIREBASE_WEB_API_KEY or FIREBASE_API_KEY is required")

    payload = json.dumps(
        {
            "email": email.strip().lower(),
            "password": password,
            "returnSecureToken": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 401}:
            raise InvalidOldPasswordError("Old password is invalid") from exc
        raise PasswordChangeError("Firebase password verification failed") from exc
    except urllib.error.URLError as exc:
        raise PasswordChangeError("Firebase password verification unavailable") from exc


def change_password(uid: str, email: str, old_password: str, new_password: str) -> None:
    clean_email = (email or "").strip().lower()
    if not clean_email:
        raise PasswordChangeError("Authenticated user email is required")
    if not old_password:
        raise InvalidOldPasswordError("Old password is required")
    if len(new_password or "") < 8:
        raise PasswordChangeError("New password must be at least 8 characters")

    _verify_email_password(clean_email, old_password)
    app = _get_password_reset_app()
    auth.update_user(uid, password=new_password, app=app)
