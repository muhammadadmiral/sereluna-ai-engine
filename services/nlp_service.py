import yake
import re
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Risk Keywords and Weights from GAS
RISK_KEYWORDS = {
    "crisis": ["bunuh diri", "mengakhiri hidup", "self harm", "menyakiti diri", "mati saja"],
    "violence": ["bunuh", "pukul", "bacok", "tusuk", "ledak"],
    "sexual": ["seks", "porno", "mesum"],
    "pii": ["nik", "ktp", "alamat lengkap", "nomor kartu"]
}
RISK_WEIGHTS = { "crisis": 3, "violence": 2, "sexual": 1, "pii": 1 }

def extract_keywords(text: str) -> List[str]:
    if not text:
        return []
    kw_extractor = yake.KeywordExtractor(lan="id", n=1, top=3, features=None)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def find_relevant_diary(current_text: str, past_diaries: List[str]) -> Optional[str]:
    if not past_diaries or not current_text:
        return None
    
    documents = past_diaries + [current_text]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    # Last vector is current_text, others are past_diaries
    current_vec = vectors[-1].reshape(1, -1)
    past_vecs = vectors[:-1]
    
    cosine_sim = cosine_similarity(current_vec, past_vecs).flatten()
    most_relevant_idx = cosine_sim.argsort()[-1]
    
    if cosine_sim[most_relevant_idx] > 0.1: # Threshold to ensure some relevance
        return past_diaries[most_relevant_idx]
    return None

def calculate_risk_level(text: str, screening_context: str = "", session_summary: str = "", client_risk: str = "") -> str:
    if client_risk:
        return client_risk
    
    combined_text = f"{screening_context} {session_summary} {text}".lower()
    
    score = 0
    for cat, keywords in RISK_KEYWORDS.items():
        for k in keywords:
            if k in combined_text:
                score += RISK_WEIGHTS[cat]
                
    crisis_pattern = re.compile(r"(bunuh diri|mengakhiri hidup|menyakiti diri|self harm)", re.IGNORECASE)
    severe_pattern = re.compile(r"(ekstrem|berat)", re.IGNORECASE)
    medium_pattern = re.compile(r"(sedang)", re.IGNORECASE)
    
    if crisis_pattern.search(combined_text):
        return "high"
    if score >= 3 or severe_pattern.search(screening_context):
        return "high"
    if score >= 2 or medium_pattern.search(screening_context):
        return "medium"
    
    return "low"
