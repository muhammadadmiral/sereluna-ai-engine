import os
from typing import Optional

from firebase_admin import auth

from services.firebase_service import initialize_firebase


def generate_password_reset_link(email: str, continue_url: Optional[str] = None) -> str:
    initialize_firebase()

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

    return auth.generate_password_reset_link(email=email.strip(), action_code_settings=action_code_settings)
