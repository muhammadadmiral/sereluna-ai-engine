import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from firebase_admin import firestore
from services.firebase_service import get_firestore_client, user_document
from services.notification_service import create_notification
from services.llm_service import _completion, _parse_json_object

def _utc_now_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def get_aura_rank(level: int) -> str:
    if level <= 5: return "New Moon Wanderer"
    elif level <= 10: return "Waxing Soul"
    elif level <= 20: return "Full Moon Guardian"
    elif level <= 35: return "Astral Sage"
    elif level <= 50: return "Celestial Navigator"
    else: return "Eternal Serenity"

def get_aura_state(level: int, streak: int, status: str, is_eclipse: bool = False) -> Dict[str, Any]:
    # Logic to determine "Vibe" of the aura
    if is_eclipse:
        return {
            "name": "Lunar Eclipse",
            "description": "Aura-mu sedang beristirahat dalam bayang-bayang. Streak-mu terlindungi oleh kegelapan yang menenangkan.",
            "color_code": "#2D3436",
            "intensity": 0.2
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

def generate_nostalgia_message(uid: str) -> Optional[str]:
    # Fetch a random old diary summary from a month ago
    # For now, let's simulate with a prompt to LLM to create a "growth" message
    # In a real app, you'd fetch real data from 'diaries' collection
    
    system_prompt = (
        "Kamu adalah Sereluna. Tugasmu adalah memberikan pesan 'Echoes of Stardust'—sebuah pesan nostalgia "
        "yang menunjukkan perkembangan emosional user. Berikan pesan yang menyentuh dan memvalidasi perjuangan mereka."
    )
    
    user_prompt = (
        "Buatlah pesan nostalgia singkat (2 kalimat) untuk user yang baru saja naik level. "
        "Pesannya harus bertema: 'Ingat sebulan lalu saat kamu merasa berat? Lihat betapa jauh kamu sudah melangkah sekarang.'"
    )
    
    try:
        content = _completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            use_fast_model=True
        )
        return content.strip()
    except Exception:
        return None

def activate_lunar_eclipse(uid: str) -> Dict[str, Any]:
    db = get_firestore_client()
    doc_ref = user_document(uid).collection("gamification").document("aura")
    
    snapshot = doc_ref.get()
    if not snapshot.exists:
        return {"success": False, "message": "Aura belum terinisialisasi."}
        
    data = snapshot.to_dict()
    stardust = data.get("stardust_balance", 0)
    
    if stardust < 50:
        return {"success": False, "message": "Stardust tidak cukup (Butuh 50)."}
        
    doc_ref.update({
        "stardust_balance": stardust - 50,
        "is_eclipse_active": True,
        "eclipse_date": _utc_now_date()
    })
    
    return {
        "success": True, 
        "message": "Lunar Eclipse aktif! Streak kamu terlindungi untuk hari ini.",
        "is_active": True
    }

def calculate_level(total_xp: int) -> int:
    if total_xp < 0: total_xp = 0
    return max(1, math.floor(math.sqrt(total_xp / 100)) + 1)

def xp_for_level(level: int) -> int:
    if level <= 1: return 0
    return ((level - 1) ** 2) * 100

def get_active_quests(uid: str, current_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Simple static quests for now, could be dynamic from DB
    today = _utc_now_date()
    streak = current_stats.get("current_streak", 0)
    
    quests = [
        {
            "id": "daily_reflection",
            "title": "Cahaya Harian",
            "description": "Tulis satu diary atau lakukan satu sesi chat hari ini.",
            "xp_reward": 50,
            "stardust_reward": 5,
            "is_completed": current_stats.get("last_activity_date") == today,
            "progress": 1.0 if current_stats.get("last_activity_date") == today else 0.0,
            "type": "daily"
        },
        {
            "id": "streak_warrior",
            "title": "Penjaga Api",
            "description": "Capai streak 3 hari berturut-turut.",
            "xp_reward": 150,
            "stardust_reward": 20,
            "is_completed": streak >= 3,
            "progress": min(1.0, streak / 3.0),
            "type": "milestone"
        },
        {
            "id": "deep_diver_weekly",
            "title": "Penyelam Jiwa",
            "description": "Tulis 3 Deep Reflection dalam seminggu.",
            "xp_reward": 300,
            "stardust_reward": 50,
            "is_completed": False, # Would need more complex tracking
            "progress": 0.3,
            "type": "weekly"
        }
    ]
    return quests

def generate_aura_reading(uid: str) -> Dict[str, Any]:
    aura = get_user_aura(uid)
    
    system_prompt = (
        "Kamu adalah Oracle Lunar Aura dari aplikasi Sereluna. "
        "Tugasmu adalah memberikan 'Aura Reading' yang puitis, dramatis, dan memotivasi layaknya narasi dalam game RPG kelas atas. "
        "Gunakan data user untuk memberikan ramalan atau pembacaan kondisi mental mereka secara metaforis. "
        "Jangan memberikan saran medis. Fokus pada pertumbuhan spiritual dan emosional."
    )
    
    user_prompt = (
        f"Data User Aura:\n"
        f"- Level: {aura['level']} ({aura['rank_title']})\n"
        f"- Streak: {aura['streak_count']} hari\n"
        f"- Status: {aura['status']}\n"
        f"- Aura State: {aura['aura_state']['name']}\n\n"
        "Buatlah pembacaan aura dalam 2-3 kalimat pendek yang sangat 'epic'. "
        "Kembalikan JSON: {\"title\": \"Judul Reading\", \"reading\": \"Isi pembacaan\", \"narrative_mood\": \"Mood narasi\"}"
    )
    
    fallback = {
        "title": "Cahaya yang Tenang",
        "reading": "Aura-mu saat ini memancarkan ketenangan. Teruslah berjalan di jalan refleksi ini.",
        "narrative_mood": "Serene"
    }
    
    try:
        content = _completion(
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
            "is_eclipse_active": False, "active_title": None
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
    is_eclipse = data.get("is_eclipse_active", False)
    today = _utc_now_date()
    
    if last_activity:
        last_date = datetime.strptime(last_activity, "%Y-%m-%d").date()
        today_date = datetime.strptime(today, "%Y-%m-%d").date()
        days_diff = (today_date - last_date).days
        
        # If eclipse was active yesterday, it doesn't count as a miss
        eclipse_date = data.get("eclipse_date")
        was_eclipsed_yesterday = False
        if eclipse_date:
            e_date = datetime.strptime(eclipse_date, "%Y-%m-%d").date()
            if (today_date - e_date).days == 1:
                was_eclipsed_yesterday = True

        if days_diff == 2 and not was_eclipsed_yesterday:
            status = "fading"
            doc_ref.update({"status": status})
        elif days_diff > 2 and not was_eclipsed_yesterday:
            current_streak = 0
            status = "active"
            doc_ref.update({"current_streak": current_streak, "status": status})

    # Reset eclipse if it's a new day
    if is_eclipse and data.get("eclipse_date") != today:
        is_eclipse = False
        doc_ref.update({"is_eclipse_active": False})

    # Get unlocked badges/titles count
    ach_doc = user_document(uid).collection("gamification").document("achievements").get()
    unlocked_badges_count = len(ach_doc.to_dict().get("unlocked_ids", [])) if ach_doc.exists else 0
    
    title_doc = user_document(uid).collection("gamification").document("titles").get()
    unlocked_titles = title_doc.to_dict().get("unlocked_titles", []) if title_doc.exists else []

    return {
        "level": level,
        "level_name": f"Level {level}",
        "rank_title": get_aura_rank(level),
        "active_title": data.get("active_title"),
        "unlocked_titles": unlocked_titles,
        "current_xp": total_xp,
        "next_level_xp": xp_for_level(level + 1),
        "progress_percentage": round(progress_percentage, 2),
        "streak_count": current_streak,
        "stardust_balance": data.get("stardust_balance", 0),
        "status": status,
        "is_eclipse_active": is_eclipse,
        "last_activity_date": last_activity,
        "aura_state": get_aura_state(level, current_streak, status, is_eclipse),
        "active_quests": get_active_quests(uid, {**data, "last_activity_date": last_activity, "current_streak": current_streak, "status": status}),
        "unlocked_badges_count": unlocked_badges_count
    }

def award_xp(uid: str, amount: int, source: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    db = get_firestore_client()
    doc_ref = user_document(uid).collection("gamification").document("aura")
    stats_ref = user_document(uid).collection("gamification").document("stats")
    
    # Celestial Multiplier
    multiplier = get_celestial_multiplier()
    final_amount = int(amount * multiplier)
    
    @firestore.transactional
    def update_in_transaction(transaction, ref, s_ref):
        snapshot = doc_ref.get(transaction=transaction)
        if not snapshot.exists:
            data = {"total_xp": 0, "level": 1, "current_streak": 0, "highest_streak": 0, "last_activity_date": None, "stardust_balance": 0, "status": "active", "is_eclipse_active": False}
        else:
            data = snapshot.to_dict() or {}
            
        stats_snapshot = s_ref.get(transaction=transaction)
        stats = stats_snapshot.to_dict() if stats_snapshot.exists else {}
            
        old_level = data.get("level", 1)
        total_xp = data.get("total_xp", 0)
        new_xp = total_xp + final_amount
        new_level = calculate_level(new_xp)
        
        last_activity = data.get("last_activity_date")
        current_streak = data.get("current_streak", 0)
        highest_streak = data.get("highest_streak", 0)
        status = data.get("status", "active")
        stardust = data.get("stardust_balance", 0)
        is_eclipse = data.get("is_eclipse_active", False)
        today = _utc_now_date()
        
        # Track Stats
        stats["total_xp_gained"] = stats.get("total_xp_gained", 0) + final_amount
        if source == "article": stats["total_articles_read"] = stats.get("total_articles_read", 0) + 1
        if source == "screening": stats["total_screenings"] = stats.get("total_screenings", 0) + 1
        if source == "chat" or source == "diary": 
            stats["total_chats"] = stats.get("total_chats", 0) + 1
            stats["total_chat_words"] = stats.get("total_chat_words", 0) + (details.get("word_count", 0) if details else 0)
        
        # Night activity (Hour 22 to 04)
        current_hour = datetime.now().hour
        if current_hour >= 22 or current_hour <= 4:
            stats["night_activities"] = stats.get("night_activities", 0) + 1

        streak_extended = False
        streak_rescued = False
        
        if last_activity != today:
            if last_activity:
                last_date = datetime.strptime(last_activity, "%Y-%m-%d").date()
                today_date = datetime.strptime(today, "%Y-%m-%d").date()
                days_diff = (today_date - last_date).days
                
                # Check for Eclipse shield
                eclipse_date = data.get("eclipse_date")
                was_eclipsed_yesterday = False
                if eclipse_date:
                    e_date = datetime.strptime(eclipse_date, "%Y-%m-%d").date()
                    if (today_date - e_date).days == 1:
                        was_eclipsed_yesterday = True

                if days_diff == 1 or was_eclipsed_yesterday:
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
        
        leveled_up = new_level > old_level
        nostalgia_msg = None
        if leveled_up:
            stardust += (new_level * 10) # Bonus currency
            nostalgia_msg = generate_nostalgia_message(uid)
            
        # Check for new titles
        stats["highest_streak"] = highest_streak
        new_titles = check_titles(uid, stats)

        updates = {
            "total_xp": new_xp,
            "level": new_level,
            "current_streak": current_streak,
            "highest_streak": highest_streak,
            "last_activity_date": today,
            "status": status,
            "stardust_balance": stardust,
            "is_eclipse_active": False # Reset on any activity
        }
        transaction.set(ref, updates, merge=True)
        transaction.set(s_ref, stats, merge=True)

        msg = "Aura-mu bersinar lebih terang!"
        if multiplier > 1.0:
            msg = "✨ CELESTIAL EVENT! XP Berlipat Ganda! ✨"

        return {
            "old_level": old_level,
            "new_level": new_level,
            "leveled_up": leveled_up,
            "xp_gained": final_amount,
            "new_total_xp": new_xp,
            "streak_extended": streak_extended,
            "streak_rescued": streak_rescued,
            "current_streak": current_streak,
            "new_titles": new_titles,
            "message": msg,
            "nostalgia_message": nostalgia_msg
        }
        
    transaction = db.transaction()
    result = update_in_transaction(transaction, doc_ref, stats_ref)
    
    # ---------------------------------------------------------
    # GAMIFICATION PUSH NOTIFICATIONS & ACHIEVEMENTS (RPG FEEL)
    # ---------------------------------------------------------
    if result["leveled_up"]:
        rank = get_aura_rank(result["new_level"])
        create_notification(
            uid=uid,
            title="🌟 LEVEL UP! 🌟",
            body=f"Selamat! Kamu mencapai Level {result['new_level']} dan meraih rank {rank}. Terus bersinar!",
            notification_type="gamification",
            priority="high",
            category_label="Level Up",
            action_link="/profile/aura"
        )
        
    if result["streak_rescued"]:
        create_notification(
            uid=uid,
            title="🛡️ STREAK RESCUED!",
            body="Deep Reflection kamu menyelamatkan streak yang hampir putus! Streak kamu aman.",
            notification_type="gamification",
            priority="high",
            category_label="Streak",
            action_link="/profile/aura"
        )
    elif result["streak_extended"] and result["current_streak"] in [3, 7, 14, 30, 50, 100]:
        create_notification(
            uid=uid,
            title="🔥 STREAK MILESTONE 🔥",
            body=f"Luar biasa! Kamu konsisten selama {result['current_streak']} hari tanpa henti.",
            notification_type="gamification",
            priority="high",
            category_label="Streak",
            action_link="/profile/aura"
        )
        
    # Check Achievements
    badge_source = "deep_diary" if (source == "diary" and details and details.get("is_deep_reflection")) else source
    unlocked_badges = check_achievements(uid, result["new_level"], result["current_streak"], badge_source)
    result["unlocked_badges"] = unlocked_badges

    return result
