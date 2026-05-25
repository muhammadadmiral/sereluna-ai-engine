import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from firebase_admin import firestore
from services.firebase_service import get_firestore_client, user_document
from services.notification_service import create_notification
from services.llm_service import _completion, _parse_json_object

def _utc_now_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def get_aura_rank(level: int, total_xp: int = 0) -> str:
    if total_xp <= 1000: return "Shadow Wanderer"
    elif total_xp <= 3000: return "Crescent Initiate"
    elif total_xp <= 6000: return "Lunar Adept"
    elif total_xp <= 10000: return "Celestial Guardian"
    else: return "Supernova Soul"

def get_tier_color(rank: str) -> str:
    colors = {
        "Shadow Wanderer": "#808080",
        "Crescent Initiate": "#ADD8E6",
        "Lunar Adept": "#6A0DAD",
        "Celestial Guardian": "#FFD700",
        "Supernova Soul": "#00FFFF"
    }
    return colors.get(rank, "#808080")

def get_aura_state(level: int, streak: int, status: str, eclipse_shields: int = 0) -> Dict[str, Any]:
    # Logic to determine "Vibe" of the aura
    if eclipse_shields > 0:
        return {
            "name": "Lunar Protection",
            "description": "Aura-mu sedang terlindungi oleh perisai gerhana. Streak-mu aman dari kegelapan.",
            "color_code": "#2D3436",
            "intensity": 0.4
        }

    if status == "fading":
        return {
            "name": "Dimming Ember",
            "description": "Aura-mu meredup karena kurangnya perhatian. Segera lakukan refleksi untuk menyalakannya kembali.",
            "color_code": "#4A4E69",
            "intensity": 0.3
        }
    
    if streak >= 30:
        return {
            "name": "Supernova Resonance",
            "description": "Aura-mu meledak dalam cahaya keemasan. Kamu berada dalam harmoni total.",
            "color_code": "#FFD700",
            "intensity": 1.0
        }
    elif streak >= 7:
        return {
            "name": "Radiant Bloom",
            "description": "Cahaya biru cerah menyelimutimu. Konsistensimu membuahkan ketenangan.",
            "color_code": "#00B4D8",
            "intensity": 0.8
        }
    else:
        return {
            "name": "Soft Glow",
            "description": "Cahaya lembut pertanda awal perjalanan yang baik.",
            "color_code": "#CAF0F8",
            "intensity": 0.5
        }

def get_celestial_multiplier() -> float:
    # Random "Celestial Event" chance (5% chance for 2x XP)
    import random
    if random.random() < 0.05:
        return 2.0
    return 1.0

def check_titles(uid: str, stats: Dict[str, Any]) -> List[str]:
    db = get_firestore_client()
    titles_ref = user_document(uid).collection("gamification").document("titles")
    doc = titles_ref.get()
    unlocked = doc.to_dict().get("unlocked_titles", []) if doc.exists else []
    
    new_titles = []
    
    # Title Definitions
    possible_titles = {
        "night_observer": {"name": "The Night Observer", "condition": lambda: stats.get("night_activities", 0) >= 5},
        "mindful_explorer": {"name": "The Mindful Explorer", "condition": lambda: stats.get("total_screenings", 0) >= 10},
        "seeker_of_truth": {"name": "Seeker of Truth", "condition": lambda: stats.get("total_chat_words", 0) >= 5000},
        "consistency_king": {"name": "Void Walker", "condition": lambda: stats.get("highest_streak", 0) >= 50},
    }
    
    for tid, tdata in possible_titles.items():
        if tdata["name"] not in unlocked and tdata["condition"]():
            new_titles.append(tdata["name"])
            unlocked.append(tdata["name"])
            
    if new_titles:
        titles_ref.set({"unlocked_titles": unlocked}, merge=True)
        # Notifications for titles
        for title in new_titles:
            create_notification(
                uid=uid,
                title="✨ New Title Unlocked! ✨",
                body=f"Kamu kini dikenal sebagai: {title}",
                notification_type="gamification",
                category_label="Title"
            )
            
    return new_titles

def activate_lunar_eclipse(uid: str) -> Dict[str, Any]:
    db = get_firestore_client()
    doc_ref = user_document(uid).collection("gamification").document("aura")
    
    snapshot = doc_ref.get()
    if not snapshot.exists:
        return {"success": False, "message": "Aura belum terinisialisasi."}
        
    data = snapshot.to_dict()
    stardust = data.get("stardust_balance", 0)
    
    if stardust < 500:
        return {"success": False, "message": "Stardust tidak cukup (Butuh 500)."}
        
    doc_ref.update({
        "stardust_balance": stardust - 500,
        "eclipse_shields": data.get("eclipse_shields", 0) + 1
    })
    
    return {
        "success": True, 
        "message": "Eclipse Shield dibeli! Streak kamu terlindungi jika kamu absen esok hari.",
        "is_active": True
    }

def calculate_level(total_xp: int) -> int:
    if total_xp < 0: total_xp = 0
    return max(1, math.floor(math.sqrt(total_xp / 100)) + 1)

def xp_for_level(level: int) -> int:
    if level <= 1: return 0
    return ((level - 1) ** 2) * 100

def get_player_card(uid: str) -> Dict[str, Any]:
    aura = get_user_aura(uid)
    rank = get_aura_rank(aura["level"], aura["current_xp"])
    
    next_tier_xp = 1001
    if aura["current_xp"] <= 1000: next_tier_xp = 1001
    elif aura["current_xp"] <= 3000: next_tier_xp = 3001
    elif aura["current_xp"] <= 6000: next_tier_xp = 6001
    elif aura["current_xp"] <= 10000: next_tier_xp = 10001
    else: next_tier_xp = aura["current_xp"]

    return {
        "tier_name": rank,
        "tier_color": get_tier_color(rank),
        "current_xp": aura["current_xp"],
        "next_tier_xp": next_tier_xp,
        "stardust": aura["stardust_balance"],
        "streak": aura["streak_count"],
        "eclipse_shields_active": aura.get("eclipse_shields", 0),
        "equipped_title": aura.get("active_title") or "Shadow Wanderer"
    }

def get_quests_list(uid: str) -> Dict[str, Any]:
    aura = get_user_aura(uid)
    today = _utc_now_date()
    
    # Simple Quest Logic
    daily = [
        {
            "id": "q1",
            "desc": "Tulis di Diary atau Chat hari ini",
            "progress": 1 if aura.get("last_activity_date") == today else 0,
            "target": 1,
            "reward_stardust": 50
        }
    ]
    
    weekly = [
        {
            "id": "w1",
            "desc": "Capai streak 3 hari",
            "progress": min(3, aura["streak_count"]),
            "target": 3,
            "reward_stardust": 300
        }
    ]
    
    return {"daily": daily, "weekly": weekly}

def generate_aura_oracle(uid: str) -> Dict[str, Any]:
    aura = get_user_aura(uid)
    
    # Fetch last 7 days of diary summaries
    db = get_firestore_client()
    diaries = user_document(uid).collection("diaries").order_by("date", direction=firestore.Query.DESCENDING).limit(7).get()
    diary_texts = [d.to_dict().get("summary", "") for d in diaries if d.to_dict().get("summary")]
    context_text = "\n".join(diary_texts) if diary_texts else "User belum banyak menulis dalam 7 hari terakhir."

    system_prompt = (
        "Kamu adalah Oracle Lunar Aura dari Sereluna. "
        "Tugasmu adalah memberikan ramalan aura yang cinematic dan puitis berdasarkan data diary user."
    )
    
    user_prompt = (
        f"Rank User: {aura['rank_title']}\n"
        f"Konteks Diary 7 Hari Terakhir:\n{context_text}\n\n"
        "Berikan pembacaan aura yang estetik (2-3 kalimat). "
        "Kembalikan JSON: {\"title\": \"Judul Reading\", \"reading\": \"Isi pembacaan\", \"narrative_mood\": \"Mood\"}"
    )
    
    fallback = {
        "title": "Cahaya yang Tenang",
        "reading": "Aura-mu saat ini memancarkan ketenangan. Teruslah berjalan di jalan refleksi ini.",
        "narrative_mood": "Serene"
    }
    
    try:
        content, provider = _completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
            use_fast_model=True
        )
        return _parse_json_object(content, fallback)
    except Exception:
        return fallback

def get_active_quests(uid: str, current_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    today = _utc_now_date()
    streak = current_stats.get("current_streak", 0)
    
    quests = [
        {
            "id": "daily_reflection",
            "title": "Cahaya Harian",
            "description": "Tulis satu diary atau lakukan satu sesi chat hari ini.",
            "xp_reward": 50,
            "stardust_reward": 50,
            "is_completed": current_stats.get("last_activity_date") == today,
            "progress": 1.0 if current_stats.get("last_activity_date") == today else 0.0,
            "type": "daily"
        }
    ]
    return quests

def check_achievements(uid: str, level: int, current_streak: int, source: str) -> List[Dict[str, str]]:
    db = get_firestore_client()
    achievements_ref = user_document(uid).collection("gamification").document("achievements")
    doc = achievements_ref.get()
    unlocked = doc.to_dict().get("unlocked_ids", []) if doc.exists else []
    
    new_achievements = []
    
    # Define Badges
    badges = {
        "first_step": {"name": "Langkah Pertama", "desc": "Menyelesaikan sesi atau bacaan pertamamu.", "condition": lambda: True},
        "streak_7": {"name": "Konsisten 7 Hari", "desc": "Mencapai streak 7 hari berturut-turut.", "condition": lambda: current_streak >= 7},
        "streak_30": {"name": "Lunar Dedication", "desc": "Mencapai streak 30 hari.", "condition": lambda: current_streak >= 30},
        "level_10": {"name": "Waxing Soul", "desc": "Mencapai Level 10.", "condition": lambda: level >= 10},
        "deep_diver": {"name": "Deep Diver", "desc": "Melakukan Deep Reflection di diary.", "condition": lambda: source == "deep_diary"},
    }
    
    for badge_id, badge in badges.items():
        if badge_id not in unlocked and badge["condition"]():
            new_achievements.append({"id": badge_id, "name": badge["name"], "desc": badge["desc"]})
            unlocked.append(badge_id)
            
    if new_achievements:
        achievements_ref.set({"unlocked_ids": unlocked}, merge=True)
        for ach in new_achievements:
            create_notification(
                uid=uid,
                title="🏆 Achievement Unlocked!",
                body=f"Kamu mendapatkan badge: {ach['name']}. {ach['desc']}",
                notification_type="gamification",
                priority="high",
                category_label="Badge",
                action_link="/profile/achievements",
                notification_key=f"achivement_{ach['id']}"
            )
            
    return new_achievements

def get_user_aura(uid: str) -> Dict[str, Any]:
    doc_ref = user_document(uid).collection("gamification").document("aura")
    snapshot = doc_ref.get()
    
    if not snapshot.exists:
        data = {
            "total_xp": 0, "level": 1, "current_streak": 0, "highest_streak": 0,
            "last_activity_date": None, "stardust_balance": 0, "status": "active",
            "eclipse_shields": 0, "active_title": None
        }
        doc_ref.set(data)
    else:
        data = snapshot.to_dict() or {}
        
    total_xp = data.get("total_xp", 0)
    level = calculate_level(total_xp)
    current_xp_in_level = total_xp - xp_for_level(level)
    xp_for_next_level = xp_for_level(level + 1) - xp_for_level(level)
    progress_percentage = (current_xp_in_level / xp_for_next_level * 100) if xp_for_next_level > 0 else 100.0
    
    last_activity = data.get("last_activity_date")
    current_streak = data.get("current_streak", 0)
    status = data.get("status", "active")
    eclipse_shields = data.get("eclipse_shields", 0)
    today = _utc_now_date()
    
    if last_activity:
        last_date = datetime.strptime(last_activity, "%Y-%m-%d").date()
        today_date = datetime.strptime(today, "%Y-%m-%d").date()
        days_diff = (today_date - last_date).days
        
        if days_diff == 2:
            if eclipse_shields > 0:
                eclipse_shields -= 1
                doc_ref.update({"eclipse_shields": eclipse_shields})
            else:
                status = "fading"
                doc_ref.update({"status": status})
        elif days_diff > 2:
            if eclipse_shields > 0:
                eclipse_shields -= 1 # Simple: one shield saves one multi-day break? No, shield should probably only save 1 day.
                # In a real app, you'd logic this better.
                doc_ref.update({"eclipse_shields": eclipse_shields})
            else:
                current_streak = 0
                status = "active"
                doc_ref.update({"current_streak": current_streak, "status": status})

    # Get unlocked badges/titles count
    ach_doc = user_document(uid).collection("gamification").document("achievements").get()
    unlocked_badges_count = len(ach_doc.to_dict().get("unlocked_ids", [])) if ach_doc.exists else 0
    
    title_doc = user_document(uid).collection("gamification").document("titles").get()
    unlocked_titles = title_doc.to_dict().get("unlocked_titles", []) if title_doc.exists else []

    return {
        "level": level,
        "level_name": f"Level {level}",
        "rank_title": get_aura_rank(level, total_xp),
        "active_title": data.get("active_title"),
        "unlocked_titles": unlocked_titles,
        "current_xp": total_xp,
        "next_level_xp": xp_for_level(level + 1),
        "progress_percentage": round(progress_percentage, 2),
        "streak_count": current_streak,
        "stardust_balance": data.get("stardust_balance", 0),
        "status": status,
        "eclipse_shields": eclipse_shields,
        "last_activity_date": last_activity,
        "aura_state": get_aura_state(level, current_streak, status, eclipse_shields),
        "active_quests": get_active_quests(uid, {**data, "last_activity_date": last_activity, "current_streak": current_streak, "status": status}),
        "unlocked_badges_count": unlocked_badges_count
    }

def award_xp(uid: str, amount: int, source: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    db = get_firestore_client()
    doc_ref = user_document(uid).collection("gamification").document("aura")
    stats_ref = user_document(uid).collection("gamification").document("stats")
    
    multiplier = get_celestial_multiplier()
    final_amount = int(amount * multiplier)
    stardust_gain = 10 # Base stardust
    
    @firestore.transactional
    def update_in_transaction(transaction, ref, s_ref):
        snapshot = doc_ref.get(transaction=transaction)
        data = snapshot.to_dict() if snapshot.exists else {"total_xp": 0, "level": 1, "current_streak": 0, "highest_streak": 0, "last_activity_date": None, "stardust_balance": 0, "status": "active", "eclipse_shields": 0}
            
        stats_snapshot = s_ref.get(transaction=transaction)
        stats = stats_snapshot.to_dict() if stats_snapshot.exists else {}
            
        old_xp = data.get("total_xp", 0)
        old_rank = get_aura_rank(data.get("level", 1), old_xp)
        
        new_xp = old_xp + final_amount
        new_rank = get_aura_rank(calculate_level(new_xp), new_xp)
        
        last_activity = data.get("last_activity_date")
        current_streak = data.get("current_streak", 0)
        highest_streak = data.get("highest_streak", 0)
        status = data.get("status", "active")
        stardust = data.get("stardust_balance", 0)
        today = _utc_now_date()
        
        # Track Stats
        stats["total_xp_gained"] = stats.get("total_xp_gained", 0) + final_amount
        if source == "article": stats["total_articles_read"] = stats.get("total_articles_read", 0) + 1
        if source == "screening": stats["total_screenings"] = stats.get("total_screenings", 0) + 1
        
        streak_extended = False
        streak_rescued = False
        
        if last_activity != today:
            if last_activity:
                last_date = datetime.strptime(last_activity, "%Y-%m-%d").date()
                today_date = datetime.strptime(today, "%Y-%m-%d").date()
                days_diff = (today_date - last_date).days
                
                if days_diff == 1:
                    current_streak += 1
                    streak_extended = True
                    status = "active"
                elif days_diff == 2 and status == "fading":
                    is_deep_reflection = source == "diary" and details and details.get("is_deep_reflection", False)
                    if is_deep_reflection:
                        current_streak += 1
                        streak_extended = True
                        streak_rescued = True
                        status = "active"
                    else:
                        current_streak = 1
                        status = "active"
                else:
                    current_streak = 1
                    status = "active"
            else:
                current_streak = 1
                streak_extended = True
                
            if current_streak > highest_streak:
                highest_streak = current_streak
        
        is_tier_up = new_rank != old_rank
        nostalgia_msg = None
        if is_tier_up:
            stardust += 100 # Tier up bonus
            nostalgia_msg = f"Luar biasa! Kamu telah mencapai tingkat kedamaian baru: {new_rank}. Teruslah merefleksikan dirimu."
            
        final_stardust = stardust + stardust_gain
        
        updates = {
            "total_xp": new_xp,
            "level": calculate_level(new_xp),
            "current_streak": current_streak,
            "highest_streak": highest_streak,
            "last_activity_date": today,
            "status": status,
            "stardust_balance": final_stardust
        }
        transaction.set(ref, updates, merge=True)
        transaction.set(s_ref, stats, merge=True)

        return {
            "is_tier_up": is_tier_up,
            "xp_gained": final_amount,
            "stardust_gained": stardust_gain,
            "new_total_xp": new_xp,
            "streak_extended": streak_extended,
            "streak_rescued": streak_rescued,
            "current_streak": current_streak,
            "celestial_event": multiplier > 1.0,
            "message": "Aura-mu makin Gacor!" if multiplier > 1.0 else "Aura-mu bersinar!",
            "nostalgia_message": nostalgia_msg
        }
        
    transaction = db.transaction()
    result = update_in_transaction(transaction, doc_ref, stats_ref)
    
    # Achievements & Oracle Echo
    if result["is_tier_up"]:
        result["oracle_echo"] = "Bulan lalu kamu khawatir... hari ini auramu jauh lebih terang. Kamu hebat."
        
    badge_source = "deep_diary" if (source == "diary" and details and details.get("is_deep_reflection")) else source
    result["unlocked_badges"] = check_achievements(uid, calculate_level(result["new_total_xp"]), result["current_streak"], badge_source)

    return result
