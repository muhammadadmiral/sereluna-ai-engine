import re
from typing import Optional


_LEADING_SUMMARY_PATTERNS = (
    r"^(?:berikut\s+(?:adalah\s+)?)?ringkasan\s+(?:dari\s+)?(?:sesi\s+)?(?:chat|percakapan)\s+sereluna(?:\s+dengan\s+[^:.\n]+)?[:.\-\s]*",
    r"^(?:berikut\s+(?:adalah\s+)?)?ringkasan\s+(?:dari\s+)?(?:sesi\s+)?(?:chat|percakapan)(?:\s+dengan\s+[^:.\n]+)?[:.\-\s]*",
    r"^(?:berikut\s+(?:adalah\s+)?)?(?:final\s+)?diary\s+summary(?:\s+dari\s+)?(?:sesi\s+)?(?:chat|percakapan)?(?:\s+sereluna)?(?:\s+dengan\s+[^:.\n]+)?[:.\-\s]*",
    r"^(?:ringkasan|summary)\s*[:.\-\s]+",
)


def clean_diary_summary(value: Optional[str], fallback: str = "") -> str:
    text = " ".join((value or "").split()).strip(" \"'")
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
