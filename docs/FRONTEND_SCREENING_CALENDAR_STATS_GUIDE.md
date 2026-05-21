# Frontend Guide: Screening, Calendar, Statistics

Backend now treats DASS-21 as a periodic baseline, while mood, diary, and sleep are daily signals.

## Recommended UX

- Do not ask users to fill DASS-21 every day.
- Check screening status first.
- Show DASS-21 as a safe screening card, not as a diagnosis.
- Use calendar and statistics endpoints for daily/periodic wellbeing UI.

Safe wording:

- "Tingkat stres minggu ini"
- "Area yang perlu diperhatikan"
- "Bukan diagnosis medis"

Avoid wording:

- "Kamu depresi"
- "Diagnosis anxiety"
- "Hasil klinis final"

## DASS-21 Flow

### 1. Get Questionnaire

```http
GET /api/v1/screening/dass21/
```

Use this before rendering the form. The questions come from:

```text
data/screening/dass21_questions.csv
```

Response shape:

```json
{
  "instrument": "DASS-21",
  "version": "bahasa_indonesia_csv_v1.0",
  "source_file": "data/screening/dass21_questions.csv",
  "recommended_interval_days": 7,
  "disclaimer": "DASS-21 adalah alat screening, bukan diagnosis medis.",
  "instructions": "Pilih jawaban 0-3 sesuai kondisi yang paling menggambarkan satu minggu terakhir.",
  "answer_options": [
    {"value": 0, "label": "Tidak pernah / tidak sesuai dengan saya"},
    {"value": 1, "label": "Kadang-kadang / sesuai sampai tingkat tertentu"},
    {"value": 2, "label": "Sering / cukup sesuai dengan saya"},
    {"value": 3, "label": "Hampir selalu / sangat sesuai dengan saya"}
  ],
  "questions": [
    {
      "id": 1,
      "category": "stress",
      "text": "...",
      "answer_min": 0,
      "answer_max": 3
    }
  ]
}
```

Frontend rule:

- Sort by `id`.
- Render 21 questions.
- Store answers as values `0..3`.
- Send answers as array of 21 numbers in question order.

### 2. Check Screening Status

```http
GET /api/v1/screening/status/
Authorization: Bearer <firebase_id_token>
```

Response:

```json
{
  "instrument": "DASS-21",
  "recommended_interval_days": 7,
  "is_due": false,
  "latest": {
    "date": "2026-05-22",
    "severity": {
      "stress": "moderate",
      "anxiety": "mild",
      "depression": "normal"
    },
    "scores": {
      "stress": 20,
      "anxiety": 8,
      "depression": 6
    },
    "summary": "..."
  },
  "next_recommended_date": "2026-05-29",
  "disclaimer": "DASS-21 adalah alat screening, bukan diagnosis medis."
}
```

Frontend rule:

- If `is_due = true`, show DASS-21 prompt/card.
- If `is_due = false`, show small baseline status or hide the prompt.
- Do not block the app if user skips screening.

### 3. Submit Screening

```http
POST /api/v1/screening/
Authorization: Bearer <firebase_id_token>
Content-Type: application/json
```

Payload:

```json
{
  "answers": [0, 1, 2, 0, 1, 3, 0, 1, 0, 2, 1, 0, 3, 1, 0, 2, 1, 0, 2, 1, 0],
  "note": "optional"
}
```

Response:

```json
{
  "date": "2026-05-22",
  "scores": {
    "depression": 12,
    "anxiety": 8,
    "stress": 18
  },
  "severity": {
    "depression": "mild",
    "anxiety": "mild",
    "stress": "mild"
  },
  "summary": "DASS-21: ...",
  "has_screening_today": true,
  "next_recommended_date": "2026-05-29",
  "recommended_interval_days": 7,
  "disclaimer": "DASS-21 adalah alat screening, bukan diagnosis medis."
}
```

## Calendar

### Month Summary

```http
GET /api/v1/calendar/summary/?year=2026&month=5
Authorization: Bearer <firebase_id_token>
```

Use this for month dots/indicators.

Important fields:

```json
{
  "items": [
    {
      "date": "2026-05-22",
      "mood": "anxious",
      "has_diary": true,
      "has_sleep_data": true,
      "wellbeing_score": 62,
      "wellbeing_level": "watch",
      "indicator": "yellow",
      "summary": "Mood cenderung tegang, tidur kurang mendukung pemulihan.",
      "recommendation": "Prioritaskan rutinitas tidur yang lebih konsisten malam ini.",
      "risk_level": "low",
      "model_version": "daily_wellbeing_v1.0",
      "screening_context": {
        "latest_date": "2026-05-20",
        "stress": "moderate",
        "anxiety": "mild",
        "depression": "normal",
        "disclaimer": "Bukan diagnosis medis."
      }
    }
  ]
}
```

Indicator mapping:

```text
green  = stable
yellow = watch
orange = attention
red    = high_attention
empty  = no data
```

### Day Detail

```http
GET /api/v1/calendar/detail/?date=2026-05-22
Authorization: Bearer <firebase_id_token>
```

Use this for the selected day detail screen.

Render:

- `summary`
- `wellbeing.score`
- `indicator`
- `wellbeing.recommendation`
- `wellbeing.signals`
- `screening_context` as baseline, not daily diagnosis

## Statistics

```http
GET /api/v1/statistics/wellbeing/?range=30d
Authorization: Bearer <firebase_id_token>
```

Alias:

```http
GET /api/v1/stats/wellbeing/?range=30d
```

Allowed ranges:

```text
7d, 30d, 90d
```

Response:

```json
{
  "range": "30d",
  "period_days": 30,
  "overall_mood": "cenderung stabil",
  "average_wellbeing_score": 74.2,
  "mood_distribution": {
    "happy": 8,
    "neutral": 12,
    "sad": 4,
    "anxious": 6,
    "angry": 0
  },
  "dominant_mood": "neutral",
  "screening_context": {
    "latest_date": "2026-05-20",
    "stress": "moderate",
    "anxiety": "mild",
    "depression": "normal",
    "disclaimer": "Bukan diagnosis medis."
  },
  "insights": [
    "Mood anxious muncul cukup sering dalam periode ini."
  ],
  "daily_items": [
    {
      "date": "2026-05-22",
      "mood": "anxious",
      "wellbeing_score": 62,
      "wellbeing_level": "watch",
      "risk_level": "low"
    }
  ],
  "model_version": "wellbeing_statistics_v1.0",
  "disclaimer": "Insight ini bukan diagnosis medis."
}
```

## Suggested Screens

### Home

- Show DASS card only when `/screening/status/` returns `is_due = true`.
- Show "Bukan diagnosis medis" under the card.

### Calendar

- Use `/calendar/summary/` for month dots.
- Use `/calendar/detail/` when user taps a date.

### Statistics

- Use `/statistics/wellbeing/?range=30d`.
- Render trend cards:
  - average wellbeing score
  - mood distribution
  - DASS baseline
  - insights

### Chat

- Do not show raw DASS score in chat bubbles.
- Chat can use backend context automatically.
- If debug mode is enabled, render `debug_metadata`, not raw clinical labels.
