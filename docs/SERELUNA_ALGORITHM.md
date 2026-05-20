# Sereluna AI Engine Algorithm Notes

Sereluna tidak hanya meneruskan pesan user ke LLM. Backend membangun beberapa sinyal NLP dan psikologi komputasional lebih dulu, lalu sinyal itu dipakai untuk mengatur respons, safety routing, memori, dan rekomendasi coping.

Pendekatan ini termasuk algoritma konvensional dan explainable NLP. Beberapa modul berbasis data lexicon CSV, dan satu modul memakai machine learning konvensional ringan: TF-IDF nearest-centroid classifier yang di-fit dari `data/lexicons/emotion_lexicon.csv`. Setiap modul punya input, proses, output, dan alasan keputusan yang bisa dilihat di `algorithm_trace`.

## Pipeline Utama

1. **Risk Classification**
   - Metode: weighted rule-based classifier.
   - Input: pesan terbaru, ringkasan screening, ringkasan sesi.
   - Output: `low`, `medium`, atau `high`, plus alasan dan evidence.
   - Fungsi: membedakan chat normal, sinyal risiko sedang, dan krisis yang perlu diarahkan ke bantuan manusia.

2. **Sentiment Scoring**
   - Metode: lexicon-based scoring bahasa Indonesia.
   - Input: pesan user dan mood signal dari aplikasi.
   - Output: skor 1-5.
   - Fungsi: memberi sinyal kasar apakah respons perlu lebih suportif, netral, atau celebratory.

3. **Diary Retrieval**
   - Metode: TF-IDF vectorization + cosine similarity.
   - Input: pesan terbaru dan ringkasan diary sebelumnya.
   - Output: diary paling relevan jika similarity melewati threshold.
   - Fungsi: membuat Sereluna bisa mengingat konteks user, bukan hanya menjawab satu pesan.

4. **Keyword Extraction**
   - Metode: YAKE keyword extraction.
   - Input: pesan user.
   - Output: kata kunci percakapan.
   - Fungsi: membantu respons tetap fokus pada topik utama.

5. **Emotion Lexicon Profiler**
   - Metode: weighted Indonesian emotion lexicon dari CSV + mood signal.
   - Output: emosi utama, intensitas, secondary emotions, dan evidence.
   - Fungsi: membedakan apakah user lebih dominan cemas, sedih, marah, lelah, malu, atau lega.

6. **TF-IDF Nearest-Centroid Emotion Classifier**
   - Metode: machine learning konvensional.
   - Training source: `data/lexicons/emotion_lexicon.csv`.
   - Proses: sistem melakukan `fit` TF-IDF character n-gram dari term emosi, membentuk centroid per kelas emosi, lalu memprediksi emosi pesan user dengan cosine similarity.
   - Fungsi: memberi pembanding ML ringan terhadap hasil lexicon scoring tanpa membutuhkan Kaggle, Jupyter, atau dataset besar.

7. **CBT-Inspired Cognitive Distortion Pattern Miner**
   - Metode: pattern matching pada teks yang sudah dinormalisasi.
   - Output: pola seperti catastrophizing, all-or-nothing thinking, mind reading, fortune telling, self-labeling, dan should statement.
   - Fungsi: mendeteksi pola pikiran yang bisa dibantu dengan reframing tanpa memberi diagnosis.

8. **Coping Pathway Decision Tree**
   - Metode: decision tree berbasis risk level, sentiment, emotion profile, distortion count, dan intent.
   - Output: pathway seperti `cbt_reframe_plus_problem_solving`, `grounding_then_plan`, `low_energy_next_step`, atau `safety_triage`.
   - Fungsi: menentukan bentuk dukungan yang paling cocok sebelum LLM membuat kalimat akhir.

9. **Adaptive Response Planner**
   - Metode: rule-based conversation planning.
   - Input: jumlah turn di room, register bahasa user, intent, risk, sentiment, dan history.
   - Output: target panjang respons, gaya bahasa, batas penggunaan nama, emoji policy, dan continuity guidance.
   - Fungsi: makin panjang room chat, respons makin santai, kontekstual, dan tidak membuka ulang seperti bot baru.

## Jawaban Singkat Jika Ditanya "Ini Cuma Hit API?"

Tidak. LLM dipakai sebagai natural language generation layer, tetapi keputusan responsnya dikontrol oleh backend Sereluna. Sebelum prompt dikirim ke LLM, sistem melakukan risk scoring, sentiment scoring, TF-IDF diary retrieval, YAKE keyword extraction, emotion profiling dari CSV lexicon, TF-IDF nearest-centroid emotion classification, cognitive distortion mining, coping pathway selection, dan adaptive response planning. Jadi output akhir LLM sudah dikondisikan oleh algoritma NLP, data mining ringan, dan machine learning konvensional yang berjalan di backend.

## Kenapa Tidak Training Model Sendiri?

Untuk domain kesehatan mental, training model deep learning sendiri butuh dataset sensitif, validasi etik, dan evaluasi safety yang kuat. Sereluna memilih pendekatan hybrid yang lebih aman untuk MVP: algoritma konvensional yang explainable untuk risk, retrieval, emotion, dan coping decision; ditambah machine learning ringan berbasis TF-IDF centroid dari CSV lexicon; lalu LLM dipakai untuk merangkai respons natural. Pendekatan ini lebih mudah diaudit karena setiap keputusan penting tetap punya trace di `algorithm_trace`.
