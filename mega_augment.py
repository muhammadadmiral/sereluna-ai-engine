import csv
import os
import time
import random
import sys
from typing import List, Dict
# Force path to find services
sys.path.append(os.getcwd())

from services.llm_service import _completion
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "training" / "emotion_dataset.csv"

EMOTIONS = ["anger", "anxiety", "fatigue", "joy", "loneliness", "neutral", "relief", "sadness", "shame"]

def generate_synthetic_data(emotion: str, count: int) -> List[str]:
    """Generates synthetic Indonesian curhatan for a specific emotion using LLM."""
    print(f"Generating {count} samples for '{emotion}'...")
    
    system_prompt = (
        "Kamu adalah psikolog klinis senior yang sedang membangun dataset besar untuk AI kesehatan mental di Indonesia. "
        "Tugasmu adalah membuat ribuan contoh curhatan yang sangat variatif, natural, dan mencerminkan emosi spesifik user. "
        "Gunakan bahasa Indonesia yang sangat beragam: dari bahasa baku, bahasa gaul Jaksel, slang daerah, hingga typo-typo natural chat."
    )
    
    user_prompt = (
        f"Buatlah {count} baris curhatan/kalimat pendek yang unik, spesifik, dan mendalam yang mencerminkan emosi '{emotion}'.\n"
        "Jangan ada pengulangan kata yang membosankan. Variasikan konteksnya (masalah kerja, cinta, keluarga, diri sendiri).\n"
        "Format: kembalikan hanya teks curhatan saja, satu baris satu curhatan. Tanpa nomor urut."
    )
    
    try:
        content, provider = _completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.95, 
            use_fast_model=True
        )
        samples = [s.strip() for s in content.split("\n") if len(s.strip()) > 5]
        return samples[:count]
    except Exception as e:
        print(f"Error generating for {emotion}: {e}")
        return []

def augment_mega():
    # 1. Read existing data to check counts
    counts = {e: 0 for e in EMOTIONS}
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"].lower()
                if label in counts:
                    counts[label] += 1

    print("Current distribution before Mega-Augment:", counts)
    
    # 2. MEGA TARGET: 1000 samples per emotion
    MEGA_TARGET = 1000
    
    for emotion in EMOTIONS:
        needed = MEGA_TARGET - counts[emotion]
        if needed <= 0:
            continue
            
        print(f"\n--- Starting Mega Augmentation for {emotion.upper()} (Target: {MEGA_TARGET}, Needed: {needed}) ---")
        
        chunk_size = 40
        while needed > 0:
            to_gen = min(chunk_size, needed)
            samples = generate_synthetic_data(emotion, to_gen)
            
            if samples:
                with open(DATASET_PATH, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["text", "label"])
                    for s in samples:
                        writer.writerow({"text": s, "label": emotion})
                
                needed -= len(samples)
                print(f"Progress for {emotion}: {MEGA_TARGET - needed}/{MEGA_TARGET}")
                time.sleep(2) # Prevent Rate Limits
            else:
                print("Failed to get samples, retrying next chunk...")
                time.sleep(5)
                
    print("\nMEGA AUGMENTATION COMPLETE!")

if __name__ == "__main__":
    augment_mega()
