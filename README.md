# 🎙️ Speech Emotion AI

**Deteksi Emosi & Intensitas dari Rekaman Suara Berbasis Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)](https://www.tensorflow.org/)

---

## 📖 Tentang Proyek

**Speech Emotion AI** adalah aplikasi web interaktif berbasis **Streamlit** yang mampu mendeteksi **emosi** dan **intensitas emosi** seseorang hanya dari rekaman suara (`.wav`). Aplikasi menggunakan model **multitask deep learning** (Keras/TensorFlow) yang memprediksi dua label sekaligus dari fitur akustik audio: jenis emosi (mis. senang, marah, sedih, netral) dan tingkat intensitasnya (mis. rendah, sedang, tinggi).

Selain prediksi utama, aplikasi juga menyediakan:
- **Timeline emosi** — membedah perubahan emosi dari detik ke detik sepanjang durasi audio.
- **Feedback loop** — pengguna dapat mengoreksi prediksi yang salah, dan koreksi tersebut disimpan sebagai data pembelajaran untuk pengembangan model selanjutnya.

---

## ✨ Fitur Utama

- 📂 **Upload Audio** — mendukung file `.wav` yang dapat langsung diputar di aplikasi.
- 🧠 **Prediksi Multitask** — satu model menghasilkan dua output sekaligus: **emosi** dan **intensitas**, lengkap dengan skor keyakinan (confidence).
- 📊 **Analisis Timeline** — menggunakan jendela geser (*sliding window* 2 detik, hop 0,5 detik) dengan penghalusan median filter untuk melacak perubahan emosi sepanjang audio (minimal durasi 3 detik).
- 🛠️ **Koreksi & Feedback** — form untuk memperbaiki prediksi yang keliru; hasil koreksi otomatis disimpan ke `database_pembelajaran_baru.csv` guna pengumpulan data logat/aksen baru.
- ⚡ **Ekstraksi Fitur MFCC** — menggunakan 40 koefisien Mel-Frequency Cepstral Coefficients (MFCC) sebagai representasi fitur audio.

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|---|---|
| Antarmuka Web | Streamlit |
| Deep Learning | TensorFlow / Keras (model multitask `.h5`) |
| Ekstraksi Fitur Audio | Librosa (MFCC) |
| Pemrosesan Sinyal | SciPy (median filter) |
| Encoding Label | Scikit-learn `LabelEncoder` (disimpan sebagai `.pkl`) |
| Manajemen Data Feedback | Pandas (CSV) |

---

## 📂 Struktur Proyek

```
speechemotion/
├── app.py                    # Aplikasi utama Streamlit
├── model_multitask.h5         # Model Keras terlatih (harus disediakan terpisah — lihat catatan di bawah)
├── le_emosi.pkl                # LabelEncoder untuk kelas emosi
├── le_intensitas.pkl           # LabelEncoder untuk kelas intensitas
└── database_pembelajaran_baru.csv  # Dibuat otomatis saat pengguna mengirim koreksi
```

> ⚠️ **Catatan:** File `model_multitask.h5` dirujuk oleh `app.py` namun **tidak disertakan** dalam repositori ini. Pastikan model tersebut ditempatkan pada direktori yang sama dengan `app.py` sebelum menjalankan aplikasi.

---

## 🚀 Instalasi & Menjalankan

### Prasyarat

- Python 3.9 atau lebih baru
- File model `model_multitask.h5` (hasil pelatihan terpisah)

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/suryaandika3005/speechemotion.git
cd speechemotion

# 2. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependensi
pip install streamlit librosa numpy scipy pandas tensorflow

# 4. Pastikan model_multitask.h5 tersedia di root folder proyek

# 5. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

---

## 🔀 Alur Penggunaan

1. Unggah file audio `.wav` melalui panel upload.
2. Tekan tombol **"🧠 Analisis Audio Sekarang"**.
3. Aplikasi menampilkan **emosi utama** dan **intensitas** hasil prediksi model.
4. Buka bagian **"📊 Lihat Detail Timeline Perubahan Emosi"** untuk melihat perubahan emosi sepanjang durasi audio (jika audio ≥ 3 detik).
5. Jika prediksi dirasa kurang tepat, buka **"🛠️ AI Salah Tebak? Berikan Koreksi"**, pilih label yang benar, lalu simpan — data ini akan digunakan untuk pengembangan model di masa depan.

---

## 🔮 Pengembangan ke Depan

- Retraining model secara berkala menggunakan data hasil feedback (`database_pembelajaran_baru.csv`).
- Dukungan format audio tambahan (mp3, m4a) melalui konversi otomatis.
- Visualisasi timeline dalam bentuk grafik interaktif, bukan hanya daftar teks.
- Deployment model yang lebih ringan untuk mempercepat waktu inferensi pada file audio panjang.

---

## 👤 Penulis

**Surya Andika**
Program Studi Informatika, Fakultas Teknologi Informasi, Universitas Andalas

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan riset di bidang pemrosesan sinyal suara dan machine learning.
