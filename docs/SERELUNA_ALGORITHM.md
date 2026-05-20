# Sereluna AI Engine Algorithm Notes

Sereluna tidak hanya meneruskan pesan user ke LLM. Backend membangun beberapa sinyal NLP dan psikologi komputasional lebih dulu, lalu sinyal itu dipakai untuk mengatur respons, safety routing, memori, dan rekomendasi coping.

Pendekatan ini termasuk algoritma konvensional, data mining ringan, explainable NLP, dan machine learning klasik. Beberapa modul berbasis data lexicon CSV. Modul ML utama memakai dataset `data/training/emotion_dataset.csv` untuk melatih TF-IDF + Logistic Regression emotion classifier, lalu dievaluasi dengan accuracy, precision, recall, F1-score, dan confusion matrix. Setiap modul punya input, proses, output, dan alasan keputusan yang bisa dilihat di `algorithm_trace`.

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

7. **TF-IDF Logistic Regression Emotion Classifier**
   - Metode: supervised machine learning klasik.
   - Training source: `data/training/emotion_dataset.csv`.
   - Proses: dataset dilatih dengan gabungan TF-IDF word n-gram, TF-IDF character n-gram, dan lexicon-score features, lalu diklasifikasi dengan Logistic Regression. Data di-split menjadi train/test, lalu dievaluasi memakai accuracy, macro precision, macro recall, macro F1, weighted F1, dan confusion matrix.
   - Fungsi: memberi prediksi emosi berbasis training data kecil yang terpisah dari lexicon agar tidak hanya rule-based.
   - Evaluasi saat ini: 180 data, 135 train, 45 test, accuracy 0.8222, macro F1 0.8321.

8. **CBT-Inspired Cognitive Distortion Pattern Miner**
   - Metode: pattern matching pada teks yang sudah dinormalisasi.
   - Output: pola seperti catastrophizing, all-or-nothing thinking, mind reading, fortune telling, self-labeling, dan should statement.
   - Fungsi: mendeteksi pola pikiran yang bisa dibantu dengan reframing tanpa memberi diagnosis.

9. **Coping Pathway Decision Tree**
   - Metode: decision tree berbasis risk level, sentiment, emotion profile, distortion count, dan intent.
   - Output: pathway seperti `cbt_reframe_plus_problem_solving`, `grounding_then_plan`, `low_energy_next_step`, atau `safety_triage`.
   - Fungsi: menentukan bentuk dukungan yang paling cocok sebelum LLM membuat kalimat akhir.

10. **Adaptive Response Planner**
   - Metode: rule-based conversation planning.
   - Input: jumlah turn di room, register bahasa user, intent, risk, sentiment, dan history.
   - Output: target panjang respons, gaya bahasa, batas penggunaan nama, emoji policy, dan continuity guidance.
   - Fungsi: makin panjang room chat, respons makin santai, kontekstual, dan tidak membuka ulang seperti bot baru.

## Jawaban Singkat Jika Ditanya "Ini Cuma Hit API?"

Tidak. LLM dipakai sebagai natural language generation layer, tetapi keputusan responsnya dikontrol oleh backend Sereluna. Sebelum prompt dikirim ke LLM, sistem melakukan risk scoring, sentiment scoring, TF-IDF diary retrieval, YAKE keyword extraction, emotion profiling dari CSV lexicon, TF-IDF nearest-centroid emotion classification, TF-IDF Logistic Regression emotion classification, cognitive distortion mining, coping pathway selection, dan adaptive response planning. Jadi output akhir LLM sudah dikondisikan oleh algoritma NLP, data mining ringan, dan machine learning konvensional yang berjalan di backend.

## Kenapa Tidak Training Model Sendiri?

Untuk domain kesehatan mental, training model deep learning sendiri butuh dataset sensitif, validasi etik, dan evaluasi safety yang kuat. Sereluna memilih pendekatan hybrid yang lebih aman untuk MVP: algoritma konvensional yang explainable untuk risk, retrieval, emotion, dan coping decision; ditambah machine learning klasik berbasis TF-IDF centroid dan TF-IDF Logistic Regression dari dataset kecil terkurasi; lalu LLM dipakai untuk merangkai respons natural. Pendekatan ini lebih mudah diaudit karena setiap keputusan penting tetap punya trace di `algorithm_trace`.

## Rumus Evaluasi ML

- Accuracy = jumlah prediksi benar / total data uji.
- Precision = TP / (TP + FP).
- Recall = TP / (TP + FN).
- F1-score = 2 * (Precision * Recall) / (Precision + Recall).

Metrik evaluasi model tersedia di `algorithm_trace.supervised_model_evaluation`.
