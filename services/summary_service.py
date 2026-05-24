import re
from typing import Dict, Optional


_LEADING_SUMMARY_PATTERNS = (
    r"^(?:berikut\s+(?:adalah\s+)?)?ringkasan\s+(?:dari\s+)?(?:sesi\s+)?(?:chat|percakapan)\s+sereluna(?:\s+dengan\s+[^:.\n]+)?[:.\-\s]*",
    r"^(?:berikut\s+(?:adalah\s+)?)?ringkasan\s+(?:dari\s+)?(?:sesi\s+)?(?:chat|percakapan)(?:\s+dengan\s+[^:.\n]+)?[:.\-\s]*",
    r"^(?:berikut\s+(?:adalah\s+)?)?(?:final\s+)?diary\s+summary(?:\s+dari\s+)?(?:sesi\s+)?(?:chat|percakapan)?(?:\s+sereluna)?(?:\s+dengan\s+[^:.\n]+)?[:.\-\s]*",
    r"^(?:ringkasan|summary)\s*[:.\-\s]+",
)


def clean_diary_summary(value: Optional[str], fallback: str = "") -> str:
    raw = value or ""
    if "#CONTENT#" in raw:
        raw = raw.rsplit("#CONTENT#", 1)[1]
    raw = re.sub(r"#TITLE#.*?(?=#CONTENT#|$)", "", raw, flags=re.IGNORECASE | re.DOTALL)
    raw = re.sub(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\d{1,2}\s+\w+\s+\d{4}\b", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", raw)
    text = " ".join(raw.split()).strip(" \"'")
    if not text:
        return fallback

    cleaned = text
    for _ in range(3):
        previous = cleaned
        for pattern in _LEADING_SUMMARY_PATTERNS:
            updated = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
            if updated != cleaned:
                cleaned = updated.strip(" \"'").lstrip(":.- ")
        if cleaned == previous:
            break

    return cleaned or fallback


def extract_diary_summary_parts(value: Optional[str]) -> Dict[str, str]:
    raw = value or ""
    title = ""
    content = ""

    title_match = re.search(r"#TITLE#\s*(.*?)(?=#CONTENT#|$)", raw, flags=re.IGNORECASE | re.DOTALL)
    content_match = re.search(r"#CONTENT#\s*(.*)$", raw, flags=re.IGNORECASE | re.DOTALL)

    if title_match:
        title = clean_diary_summary(title_match.group(1))
    if content_match:
        content = clean_diary_summary(content_match.group(1))

    if not content:
        content = clean_diary_summary(raw)
    if not title:
        title_source = re.sub(r"^(?:User|Sereluna)\s*:\s*", "", content).strip()
        title = title_source[:48].rstrip() + ("..." if len(title_source) > 48 else "")
    if not title:
        title = "Sesi Percakapan"

    return {"title": title, "content": content}
