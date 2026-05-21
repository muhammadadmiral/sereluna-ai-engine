# 🌕 GAMIFICATION MASTERPIECE: "LUNAR AURA" 🌕
## Frontend Implementation Guide (The "Last Dance" Edition)

This document outlines how to transform the gamification system into a "killer feature" that feels like a premium RPG experience.

---

### 1. THE "AURA" UI CONCEPT
Instead of a simple progress bar, we want the user's avatar to be surrounded by a dynamic, breathing "Aura".

- **The Aura Ring:** A circular gradient glow around the profile picture.
- **Dynamic Colors (from `aura_state.color_code`):**
    - `Soft Glow`: Gentle pulsing light blue.
    - `Radiant Bloom`: Bright cyan with particle effects (sparkles).
    - `Supernova Resonance`: Golden/Amber glow with "flame" or "orbiting light" animations.
    - `Dimming Ember`: Desaturated, slow-pulsing gray/purple.
- **Intensity:** Use `aura_state.intensity` to control the opacity or speed of the pulse animation.

### 2. QUEST SYSTEM (NEW!)
Endpoint: `GET /api/v1/gamification` (included in `active_quests`)

Display these in a "Quests" or "Daily Tasks" tab:
- **Daily Quests:** Small cards with a checkmark.
- **Weekly/Milestones:** Larger cards with progress bars (`quest.progress`).
- **Interaction:** When a quest is completed, show a "Claim Reward" animation where Stardust particles fly towards the user's balance.

### 3. ORACLE READINGS (The LLM "Gacor" Feature)
Endpoint: `POST /api/v1/gamification/reading`

This is a cinematic experience. 
- **The UI:** A "Moon Oracle" screen with a crystal ball or a moon background.
- **The Reveal:** Text should appear letter-by-letter (typewriter effect).
- **The Vibe:** Use `narrative_mood` to change the background music or background color intensity.

### 4. PUSH NOTIFICATIONS & TOASTS
When a user finishes a chat or screening, the response includes a `gamification` object.

**Frontend Action:**
- **XP Gained:** Show a floating text animation `+35 XP` rising from the "Finish" button.
- **Level Up:** Full-screen overlay! Confetti, sound effect, and the new Rank Title shown in big, bold gold letters.
- **Streak Rescue:** A "Shield" icon appearing and breaking the "Fading" state.

### 5. TECHNICAL INTEGRATION STEPS

#### A. Global Gamification Listener
Create a global state/listener for the `gamification` field in all POST responses (Chat Finish, Screening, Article Read). 
If `gamification` is present, trigger the appropriate UI animations.

#### B. Aura Animation (CSS/Flutter/React Native)
Use a library like `Lottie` or custom shaders for the Aura. 
Example CSS Logic:
```css
.aura-glow {
  box-shadow: 0 0 20px 10px var(--aura-color);
  animation: pulse calc(2s / var(--aura-intensity)) infinite;
}
```

#### C. Model Routing Reminder
The backend now uses "Fast Models" (Llama 3.1 8B) for summaries and aura readings. This means responses are near-instant. Don't be afraid to call the `reading` endpoint; it's designed to be snappy.

---

### 🚀 NEW: THE "COOLER" FEATURES (GACOR EDITION) 🚀

#### 1. LUNAR ECLIPSE (Streak Freeze)
Endpoint: `POST /api/v1/gamification/eclipse`
- **Cost:** 50 Stardust.
- **UI:** User clicks a "Shield" icon. The Aura turns into a dark, void-like gray (`#2D3436`).
- **Effect:** If the user misses a day, their streak is NOT reset. It's a "Rest Day" for mental health.

#### 2. CONSTELLATION TITLES
Endpoint: `POST /api/v1/gamification/title/{title_name}`
- **Concept:** Dynamic titles that appear under the user's name.
- **Examples:** 
    - `The Night Observer`: Unlocked if active late at night.
    - `The Mindful Explorer`: Unlocked after many screenings.
    - `Seeker of Truth`: Unlocked after long chat sessions.
- **UI:** Display as a golden badge or small text banner under the avatar.

#### 3. ECHOES OF STARDUST (Nostalgia)
Included in: `GamificationUpdateResponse` (on Level Up)
- **Concept:** An emotional narrative reward.
- **Content:** A message like *"Look how far you've come. A month ago, you were worried about X... and you survived it."*
- **UI:** A floating parchment or a "Letter from the Past" animation that pops up during the Level Up screen.

#### 4. CELESTIAL EVENTS
- **Concept:** Random 5% chance of getting `2x XP`.
- **UI:** If `multiplier` in the backend response is > 1.0, show a "CELESTIAL EVENT" banner with shooting stars across the screen.

---

### 🚀 LET'S MAKE THIS A MASTERPIECE 🚀
The "Last Dance" is about polish. Every level up should feel like a victory. Every streak rescue should feel like a relief. The user isn't just journaling; they are **leveling up their soul**.
