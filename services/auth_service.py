import json
import os
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
