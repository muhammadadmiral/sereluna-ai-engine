from typing import Any, Dict, List, Optional

from pydantic import BaseModel

class Quest(BaseModel):
    id: str
    title: str
    description: str
    xp_reward: int
    stardust_reward: int
    is_completed: bool
    progress: float  # 0.0 to 1.0
    type: str  # "daily", "weekly", "milestone"

class AuraState(BaseModel):
    name: str
    description: str
    color_code: str  # Hex code for UI
    intensity: float  # 0.0 to 1.0

class GamificationAuraResponse(BaseModel):
    level: int
    level_name: str
    rank_title: str
    active_title: Optional[str] = None
    unlocked_titles: List[str] = []
    current_xp: int
    next_level_xp: int
    progress_percentage: float
    streak_count: int
    stardust_balance: int
    status: str
    is_eclipse_active: bool = False
    last_activity_date: Optional[str]
    aura_state: AuraState
    active_quests: List[Quest] = []
    unlocked_badges_count: int = 0

class GamificationUpdateResponse(BaseModel):
    xp_gained: int
    stardust_gained: int = 0
    new_total_xp: int
    streak_extended: bool
    streak_rescued: bool
    is_tier_up: bool = False
    celestial_event: bool = False
    new_level: int = 1
    unlocked_badges: List[Dict[str, str]] = []
    new_titles: List[str] = []
    message: str
    quest_completed: List[Quest] = []
    nostalgia_message: Optional[str] = None
    oracle_echo: Optional[str] = None

class PlayerCardResponse(BaseModel):
    tier_name: str
    tier_color: str
    current_xp: int
    next_tier_xp: int
    stardust: int
    streak: int
    eclipse_shields_active: int
    equipped_title: Optional[str] = None

class QuestItem(BaseModel):
    id: str
    desc: str
    progress: int
    target: int
    reward_stardust: int

class QuestsListResponse(BaseModel):
    daily: List[QuestItem]
    weekly: List[QuestItem]

class AuraReadingResponse(BaseModel):
    reading: str
    title: str
    narrative_mood: str

class EclipseResponse(BaseModel):
    success: bool
    message: str
    is_active: bool
