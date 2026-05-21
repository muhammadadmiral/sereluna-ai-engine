from typing import Any, Dict

from fastapi import APIRouter, Depends

from schemas.gamification_schema import GamificationAuraResponse, AuraReadingResponse, EclipseResponse
from services.firebase_service import get_current_user, user_document
from services.gamification_service import get_user_aura, generate_aura_reading, activate_lunar_eclipse

router = APIRouter(prefix="/api/v1/gamification", tags=["gamification"])

@router.get("/", response_model=GamificationAuraResponse)
@router.get("", response_model=GamificationAuraResponse, include_in_schema=False)
async def read_user_gamification(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    aura_data = get_user_aura(current_user["uid"])
    return GamificationAuraResponse(**aura_data)

@router.post("/reading", response_model=AuraReadingResponse)
async def get_my_aura_reading(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Generate a poetic, RPG-style reading of the user's current aura.
    Uses LLM (fast model) to create a narrative experience.
    """
    reading_data = generate_aura_reading(current_user["uid"])
    return AuraReadingResponse(**reading_data)

@router.post("/eclipse", response_model=EclipseResponse)
async def trigger_lunar_eclipse(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Activate Lunar Eclipse (Streak Freeze).
    Costs 50 Stardust.
    """
    result = activate_lunar_eclipse(current_user["uid"])
    return EclipseResponse(**result)

@router.post("/title/{title_name}")
async def set_active_title(
    title_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Set the user's active constellation title.
    User must have unlocked the title first.
    """
    uid = current_user["uid"]
    title_doc = user_document(uid).collection("gamification").document("titles").get()
    unlocked = title_doc.to_dict().get("unlocked_titles", []) if title_doc.exists else []
    
    if title_name not in unlocked:
        return {"success": False, "message": "Gelar belum dibuka."}
        
    user_document(uid).collection("gamification").document("aura").update({
        "active_title": title_name
    })
    
    return {"success": True, "message": f"Gelar diatur menjadi: {title_name}", "active_title": title_name}
