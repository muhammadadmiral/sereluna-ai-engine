from pydantic import BaseModel


class MoodDailyRequest(BaseModel):
    date: str
    mood: str


class MoodDailyResponse(BaseModel):
    success: bool = True
