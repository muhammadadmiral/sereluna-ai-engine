from fastapi import APIRouter, HTTPException, status
from firebase_admin.auth import EmailNotFoundError, UserNotFoundError

from schemas.auth_schema import ForgotPasswordRequest, ForgotPasswordResponse
from services.auth_service import generate_password_reset_link

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
