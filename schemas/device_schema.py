from pydantic import BaseModel


class DeviceTokenRequest(BaseModel):
    token: str


class DeviceTokenResponse(BaseModel):
    success: bool = True
