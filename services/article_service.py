import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

from fastapi import HTTPException, status


GUARDIAN_SEARCH_URL = "https://content.guardianapis.com/search"
DEFAULT_SECTION = None
DEFAULT_TOPIC = "wellbeing"

MENTAL_HEALTH_TOPICS: Dict[str, Dict[str, Any]] = {
    "wellbeing": {
        "label": "Wellbeing",
        "query": '"mental health" OR wellbeing OR "emotional wellbeing" OR mindfulness',
        "summary": "Artikel umum tentang kesehatan mental, kebiasaan sehat, dan wellbeing.",
    },
    "stress": {
        "label": "Stres",
        "query": '"mental health" AND (stress OR burnout OR pressure OR overwhelmed)',
        "summary": "Artikel untuk memahami stres, burnout, tekanan, dan pemulihan.",
    },
    "anxiety": {
        "label": "Cemas",
        "query": '"mental health" AND (anxiety OR anxious OR worry OR panic)',
        "summary": "Artikel tentang kecemasan, kekhawatiran, dan cara mengelola rasa tegang.",
    },
    "sleep": {
        "label": "Tidur",
        "query": '(sleep OR insomnia) AND ("mental health" OR wellbeing OR stress)',
        "summary": "Artikel tentang tidur, insomnia, istirahat, dan kaitannya dengan wellbeing.",
    },
    "mood": {
        "label": "Mood",
        "query": 'mood AND ("mental health" OR wellbeing OR depression OR anxiety)',
        "summary": "Artikel tentang perubahan mood dan pola emosi sehari-hari.",
    },
    "loneliness": {
        "label": "Kesepian",
        "query": '(loneliness OR lonely OR isolation) AND ("mental health" OR wellbeing)',
        "summary": "Artikel tentang kesepian, dukungan sosial, dan koneksi emosional.",
    },
    "self-care": {
        "label": "Self-care",
        "query": '("self care" OR "self-care" OR mindfulness) AND ("mental health" OR wellbeing)',
        "summary": "Artikel tentang self-care, rutinitas kecil, dan pemulihan harian.",
    },
}

MOOD_TOPIC_MAP = {
    "happy": "wellbeing",
    "neutral": "wellbeing",
    "sad": "mood",
    "anxious": "anxiety",
    "angry": "stress",
}

MENTAL_HEALTH_KEYWORDS = {
    "mental health",
    "wellbeing",
    "well-being",
    "stress",
    "burnout",
    "anxiety",
    "anxious",
    "sleep",
    "insomnia",
    "mindfulness",
    "mood",
    "depression",
    "loneliness",
    "self-care",
    "therapy",
    "psychology",
}


def get_article_topics() -> Dict[str, Any]:
    return {
        "default_topic": DEFAULT_TOPIC,
        "topics": [
            {
                "key": key,
                "label": value["label"],
                "summary": value["summary"],
            }
            for key, value in MENTAL_HEALTH_TOPICS.items()
        ],
        "mood_topic_map": MOOD_TOPIC_MAP,
    }


def _guardian_api_key() -> str:
    api_key = (os.getenv("GUARDIAN_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GUARDIAN_API_KEY is not configured on the backend",
        )
    return api_key


def _safe_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()


def _topic_from_inputs(topic: Optional[str], mood: Optional[str]) -> str:
    normalized_topic = (topic or "").strip().lower()
    if normalized_topic in MENTAL_HEALTH_TOPICS:
        return normalized_topic
    normalized_mood = (mood or "").strip().lower()
    return MOOD_TOPIC_MAP.get(normalized_mood, DEFAULT_TOPIC)


def _relevance_score(article: Dict[str, Any], topic_key: str) -> int:
    text = " ".join(
        str(article.get(field) or "")
        for field in ("title", "summary", "section")
    ).lower()
    score = sum(1 for keyword in MENTAL_HEALTH_KEYWORDS if keyword in text)
    topic_label = MENTAL_HEALTH_TOPICS.get(topic_key, {}).get("label", "").lower()
    if topic_key in text or topic_label in text:
        score += 2
    return score


def _why_recommended(article: Dict[str, Any], topic_key: str, mood: Optional[str]) -> str:
    topic = MENTAL_HEALTH_TOPICS.get(topic_key, MENTAL_HEALTH_TOPICS[DEFAULT_TOPIC])
    mood_text = f" dan mood {mood}" if mood else ""
    return f"Relevan untuk tema {topic['label'].lower()}{mood_text}; artikel ini bisa jadi bacaan pendukung, bukan nasihat medis."


def _extract_tags(item: Dict[str, Any]) -> List[str]:
    tags = item.get("tags") or []
    return [tag.get("webTitle") or "" for tag in tags if tag.get("webTitle")][:5]


def _extract_article(item: Dict[str, Any], topic_key: str, mood: Optional[str]) -> Dict[str, Any]:
    fields = item.get("fields") or {}
    article = {
        "id": item.get("id") or "",
        "title": item.get("webTitle") or "",
        "section": item.get("sectionName") or "",
        "published_at": item.get("webPublicationDate") or "",
        "url": item.get("webUrl") or "",
        "api_url": item.get("apiUrl") or "",
        "summary": _strip_html(fields.get("trailText") or ""),
        "thumbnail": fields.get("thumbnail") or "",
        "tags": _extract_tags(item),
        "source": "The Guardian",
        "topic": topic_key,
        "topic_label": MENTAL_HEALTH_TOPICS.get(topic_key, MENTAL_HEALTH_TOPICS[DEFAULT_TOPIC])["label"],
        "content_type": "external_article",
        "content_warning": "Artikel eksternal. Gunakan sebagai bacaan pendukung, bukan diagnosis atau pengganti bantuan profesional.",
    }
    article["relevance_score"] = _relevance_score(article, topic_key)
    article["why_recommended"] = _why_recommended(article, topic_key, mood)
    return article


def search_guardian_articles(
    query: Optional[str] = None,
    topic: Optional[str] = None,
    mood: Optional[str] = None,
    limit: int = 5,
    section: Optional[str] = DEFAULT_SECTION,
) -> Dict[str, Any]:
    safe_limit = _safe_int(limit, 1, 10)
    topic_key = _topic_from_inputs(topic, mood)
    topic_payload = MENTAL_HEALTH_TOPICS[topic_key]
    safe_query = (query or topic_payload["query"]).strip()
    params = {
        "api-key": _guardian_api_key(),
        "q": safe_query,
        "page-size": min(20, max(safe_limit * 2, safe_limit)),
        "order-by": "newest",
        "show-fields": "trailText,thumbnail",
        "show-tags": "keyword",
    }
    if section:
        params["section"] = section

    url = f"{GUARDIAN_SEARCH_URL}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch articles from The Guardian",
        ) from exc

    response_payload = payload.get("response") or {}
    results = response_payload.get("results") or []
    articles = [_extract_article(item, topic_key, mood) for item in results]
    articles.sort(key=lambda item: (item["relevance_score"], item["published_at"]), reverse=True)
    articles = articles[:safe_limit]
    return {
        "source": "The Guardian",
        "query": safe_query,
        "topic": topic_key,
        "topic_label": topic_payload["label"],
        "topic_summary": topic_payload["summary"],
        "mood": mood,
        "section": section,
        "count": len(articles),
        "disclaimer": "Artikel berasal dari The Guardian dan hanya untuk edukasi ringan, bukan diagnosis medis.",
        "articles": articles,
    }

def article_notification_body(article: Dict[str, Any]) -> str:
    title = article.get("title") or "Ada artikel rekomendasi baru"
    return f"Artikel rekomendasi dari The Guardian: {title}"
