from typing import Optional

from pydantic import BaseModel


class ForgotPasswordRequest(BaseModel):
    email: str
    continue_url: Optional[str] = None


class ForgotPasswordResponse(BaseModel):
    message: str
    reset_link: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class MessageResponse(BaseModel):
    message: str
