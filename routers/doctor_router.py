from typing import Any, Dict

from fastapi import APIRouter, Depends

from schemas.doctor_schema import DoctorListResponse
from services.doctor_service import list_doctors
from services.firebase_service import get_current_user

router = APIRouter(prefix="/doctors", tags=["doctors"])


@router.get("/", response_model=DoctorListResponse)
@router.get("", response_model=DoctorListResponse, include_in_schema=False)
async def read_doctors(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return DoctorListResponse(doctors=list_doctors())
