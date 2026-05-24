import re
from typing import Optional


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
