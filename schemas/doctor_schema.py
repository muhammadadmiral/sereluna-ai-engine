from typing import List, Optional

from pydantic import BaseModel


class Doctor(BaseModel):
    id: str
    name: str
    specialty: str
    whatsapp_number: str
    image_url: Optional[str] = ""


class DoctorListResponse(BaseModel):
    doctors: List[Doctor]
