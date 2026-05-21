from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query

from schemas.article_schema import (
    ArticleNotificationRequest,
    ArticleNotificationResponse,
    ArticleRecommendationResponse,
    ArticleTopicResponse,
)
from services.article_service import article_notification_body, get_article_topics, search_guardian_articles
from services.firebase_service import get_current_user
from services.notification_service import create_article_recommendation_notification

router = APIRouter(prefix="/articles", tags=["articles"])


@router.get("/topics/", response_model=ArticleTopicResponse)
@router.get("/topics", response_model=ArticleTopicResponse, include_in_schema=False)
async def read_article_topics(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return get_article_topics()


@router.get("/recommendations/", response_model=ArticleRecommendationResponse)
@router.get("/recommendations", response_model=ArticleRecommendationResponse, include_in_schema=False)
async def read_article_recommendations(
    query: Optional[str] = Query(None, min_length=2, max_length=120),
    topic: Optional[str] = Query(None, max_length=40),
    mood: Optional[str] = Query(None, max_length=40),
    limit: int = Query(5, ge=1, le=10),
    section: Optional[str] = Query(None, max_length=80),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return search_guardian_articles(query=query, topic=topic, mood=mood, limit=limit, section=section)


@router.post("/recommendations/notify/", response_model=ArticleNotificationResponse)
@router.post("/recommendations/notify", response_model=ArticleNotificationResponse, include_in_schema=False)
async def create_article_notification(
    request: ArticleNotificationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    article = {"title": request.title, "summary": request.summary or ""}
    notification_id = create_article_recommendation_notification(
        uid=current_user["uid"],
        title="Artikel rekomendasi",
        body=article_notification_body(article),
        action_link=request.url,
        article_id=request.article_id,
    )
    return ArticleNotificationResponse(notification_id=notification_id)
