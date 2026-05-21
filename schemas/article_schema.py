from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ArticleItem(BaseModel):
    id: str
    title: str = ""
    section: str = ""
    published_at: str = ""
    url: str = ""
    api_url: str = ""
    summary: str = ""
    thumbnail: str = ""
    tags: List[str] = Field(default_factory=list)
    source: str = "The Guardian"
    topic: str = ""
    topic_label: str = ""
    content_type: str = "external_article"
    content_warning: str = ""
    relevance_score: int = 0
    why_recommended: str = ""


class ArticleRecommendationResponse(BaseModel):
    source: str = "The Guardian"
    query: str
    topic: str
    topic_label: str
    topic_summary: str
    mood: Optional[str] = None
    section: Optional[str] = None
    count: int = 0
    disclaimer: str = ""
    articles: List[ArticleItem] = Field(default_factory=list)


class ArticleTopicResponse(BaseModel):
    default_topic: str
    topics: List[Dict[str, Any]]
    mood_topic_map: Dict[str, str]


class ArticleNotificationRequest(BaseModel):
    article_id: str
    title: str
    url: str
    summary: Optional[str] = ""


class ArticleNotificationResponse(BaseModel):
    success: bool = True
    notification_id: str
