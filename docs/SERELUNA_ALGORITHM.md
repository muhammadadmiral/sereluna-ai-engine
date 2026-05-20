# Sereluna AI Engine Algorithm Notes (Thesis Edition)

Sereluna tidak hanya meneruskan pesan user ke LLM. Backend membangun beberapa sinyal NLP dan psikologi komputasional lebih dulu, lalu sinyal itu dipakai untuk mengatur respons, safety routing, memori, dan rekomendasi coping. Pendekatan ini dirancang untuk memenuhi standar akademis skripsi IT dengan fokus pada **Explainability**, **Hybrid NLP**, dan **Proactive Mental Health Support**.

Setiap modul memiliki input, proses, output, dan alasan keputusan yang bisa dilihat di `algorithm_trace` (konsep **Explainable AI / XAI**).

## Arsitektur Hybrid NLP

Sereluna menggabungkan tiga pendekatan utama:
1.  **Rule-based & Lexicon:** Untuk akurasi tinggi pada kata-kata kunci sensitif (Risk & Crisis).
2.  **Machine Learning Klasik (TF-IDF + Logistic Regression):** Untuk fleksibilitas dalam menangani variasi bahasa gaul/slang Indonesia yang dinamis.
3.  **Generative AI (LLM):** Sebagai layer Natural Language Generation (NLG) yang sudah terkondisi oleh sinyal algoritma backend.

## Pipeline & Algoritma Utama

### 1. Risk & Crisis Classification (Weighted Rule-Based)
Mendeteksi sinyal bahaya (bunuh diri, kekerasan, dsb) menggunakan pembobotan kata kunci dari `risk_patterns.csv`. Outputnya adalah `risk_level` yang menentukan apakah user butuh bantuan manusia segera.

### 2. Obfuscation Filter (Deterministic Preprocessing)
Mendeteksi kata-kata sensitif yang disamarkan (misal: "b.u.n.u.h", "4nj1ng"). Menggunakan kombinasi Leetspeak Normalization dan Levenshtein Fuzzy Matching.
- **Contextual Slang Handling:** Algoritma ini memiliki logika khusus untuk mendeteksi konteks positif (seperti "wkwk" atau "gokil") guna mencegah "False Positive" pada penggunaan slang intensifier.

### 3. Hybrid Emotion Classification (Lexicon vs. Supervised ML)
Ini adalah inti dari **Comparative Analysis** dalam skripsi:
- **Lexicon Profiler:** Mengukur emosi berdasarkan kamus kata (`emotion_lexicon.csv`). Sangat akurat untuk kata emosi yang eksplisit.
- **Logistic Regression Classifier:** Melatih model dengan dataset `emotion_dataset.csv` menggunakan fitur TF-IDF (N-Grams). Sangat efektif untuk menangkap pola kalimat informal dan slang yang tidak ada di kamus.
- **Explainability:** Sistem mencatat "evidence" (bukti kata) dan skor probabilitas untuk setiap prediksi.

### 4. Implicit DASS-21 Prediction (Proactive Screening)
Fitur inovatif untuk skripsi: Sistem memetakan emosi, intensitas, dan distorsi kognitif user ke dalam kategori DASS-21 (Depression, Anxiety, Stress) secara implisit.
- **Tujuan:** Memberikan peringatan dini (early warning) jika user menunjukkan tren gejala gangguan mental tanpa harus mengisi form tes yang panjang setiap saat.

### 5. CBT-Inspired Cognitive Distortion Mining
Mendeteksi pola pikir tidak sehat (seperti *catastrophizing* atau *overgeneralization*) menggunakan teknik Pattern Matching. Hasil deteksi ini digunakan untuk memberikan arahan "Reframing" pada respons AI.

### 6. Coping Pathway Decision Tree
Menentukan strategi dukungan terbaik berbasis logika pohon keputusan. Menggabungkan input dari Risk, Emotion, dan Cognitive Distortion untuk memilih jalur intervensi (misal: Grounding, Emotional Validation, atau Problem Solving).

### 7. Adaptive Response Planner
Mengatur gaya bahasa AI secara dinamis:
- **User Register:** Menyesuaikan penggunaan "aku-kamu" atau "gua-lu" berdasarkan input user.
- **Relationship Stage:** Semakin sering chat, respons AI semakin santai dan kontekstual (mengurangi pembukaan formal).

## Jawaban Skripsi: "Kenapa Pakai Hybrid, Bukan Deep Learning Saja?"

1.  **Explainability:** Dalam kesehatan mental, kita harus tahu *kenapa* AI mengambil keputusan tertentu. Model hybrid kita jauh lebih mudah diaudit daripada model Black Box (Deep Learning).
2.  **Efficiency:** Model TF-IDF + Logistic Regression sangat ringan, cepat, dan bisa dilatih dengan dataset kecil namun berkualitas (curated dataset).
3.  **Safety:** Menghindari halusinasi LLM dengan memberikan batasan instruksi (constraints) yang ketat berdasarkan hasil perhitungan algoritma deterministik di backend.

## Metrik Evaluasi Model (Bab 4 Skripsi)

Model supervised ML kita dievaluasi menggunakan metrik formal:
- **Accuracy:** Seberapa sering prediksi emosi benar secara keseluruhan.
- **Precision & Recall:** Penting untuk memastikan emosi negatif (seperti Sadness) tidak terlewatkan (Recall) dan tidak salah tebak (Precision).
- **F1-Score:** Keseimbangan antara Precision dan Recall.

Data evaluasi real-time tersedia di objek `supervised_model_evaluation` dalam setiap respon sistem.
