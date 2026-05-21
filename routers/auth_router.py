from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from firebase_admin.auth import EmailNotFoundError, UserNotFoundError

from schemas.auth_schema import ChangePasswordRequest, ForgotPasswordRequest, ForgotPasswordResponse, MessageResponse
from services.auth_service import (
    InvalidOldPasswordError,
    PasswordAuthConfigError,
    PasswordChangeError,
    change_password,
    generate_password_reset_link,
)
from services.firebase_service import get_current_user

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/forgot-password/", response_model=ForgotPasswordResponse)
async def forgot_password(request: ForgotPasswordRequest):
    email = request.email.strip().lower()
    continue_url = request.continue_url.strip() if request.continue_url else None

    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="email is required")

    try:
        reset_link = generate_password_reset_link(email=email, continue_url=continue_url)
    except (EmailNotFoundError, UserNotFoundError):
        return ForgotPasswordResponse(
            message="If the email exists, a password reset link has been generated.",
            reset_link=None,
        )

    return ForgotPasswordResponse(
        message="Password reset link generated successfully.",
        reset_link=reset_link,
    )


@router.post("/change-password", response_model=MessageResponse)
@router.post("/change-password/", response_model=MessageResponse, include_in_schema=False)
async def update_password(
    request: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    claims = current_user.get("claims") or {}
    firebase_claims = claims.get("firebase") or {}
    provider = firebase_claims.get("sign_in_provider") or ""
    if provider != "password":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password can only be changed for email/password accounts",
        )

    try:
        change_password(
            uid=current_user["uid"],
            email=current_user.get("email", ""),
            old_password=request.old_password,
            new_password=request.new_password,
        )
    except InvalidOldPasswordError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except PasswordAuthConfigError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except PasswordChangeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return MessageResponse(message="Password changed successfully.")
