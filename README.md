# Sistem Otomatisasi Verifikasi dan Kelayakan Pas Foto Berbasis AI

Sistem ini merupakan kerangka kerja (*framework*) berbasis *Computer Vision* dan *Machine Learning* yang dirancang untuk memverifikasi serta menilai kelayakan pas foto formal secara otomatis. Sistem ini menggabungkan pendekatan *Deep Feature Extraction* dengan *One-Class Classification* dan analisis heuristik citra digital (*rule-based image processing*).

---

## 1. Pendekatan dan Arsitektur Teknis

Sistem menggunakan pendekatan hibrida (*hybrid approach*) yang menggabungkan kekuatan representasi fitur berbasis *deep learning* dan kecepatan komputasi pengolahan citra digital klasik:

1. **Feature Extractor (ResNet50 Pre-trained)**:
   - Menggunakan arsitektur ResNet50 yang telah dilatih pada ImageNet (*transfer learning*).
   - Lapisan klasifikasi akhir (*fully connected layer*) dilepas (*headless model*), menghasilkan vektor representasi visual berdimensi 2048.
2. **One-Class Classification (One-Class SVM)**:
   - Menggunakan kernel RBF (*Radial Basis Function*) untuk memetakan distribusi pola pas foto yang memenuhi syarat (*approved*).
   - Dipilih karena dalam studi kasus penerimaan dokumen/foto formal, kelas anomali (*rejected*) bersifat *open-set* (variasi kesalahan tidak terbatas dan tidak seimbang). Dengan demikian, model hanya memodelkan batas keputusan (*decision boundary*) dari kelas positif (*approved*).
3. **Analisis Heuristik Citra Digital (Rule-Based Validation)**:
   - **Deteksi Watermark & Teks Bawah**: Analisis morfologi biner dan *connected components* untuk mendeteksi teks stempel, nama sekolah, atau *banner* identitas di bagian bawah foto.
   - **Validasi Rasio Aspek (Aspect Ratio)**: Verifikasi rasio potret standar 4:3 (toleransi deviasi ±10%).
   - **Deteksi Wajah**: Memanfaatkan Haar Cascade classifier untuk memastikan keberadaan wajah tunggal yang proporsional.
   - **Analisis Ketajaman (Blur Detection)**: Menghitung variansi operator Laplacian (skor rendah mengindikasikan citra buram/tidak fokus).
   - **Analisis Pencahayaan (Brightness)**: Menghitung rata-rata intensitas *luminance* pada ruang warna keabuan (*grayscale*) untuk mendeteksi *underexposure* atau *overexposure*.
   - **Keseragaman Latar Belakang (Background Uniformity)**: Menghitung variansi dan deviasi warna pada area tepi (*border*) atas, kiri, dan kanan.

---

## 2. Struktur Direktori dan Dataset

Dataset diorganisasikan ke dalam dua direktori utama: `raw` (citra asli dengan resolusi variatif) dan `processed` (citra hasil standardisasi ukuran 300x400 piksel format RGB).

```text
dataset/
├── raw/
│   ├── training/
│   │   └── approved-penerimaan/   # Data latih pas foto approved asli
│   ├── validation/
│   │   └── approved/              # Data validasi approved asli (untuk tuning threshold)
│   └── testing/
│       ├── approved/              # Data uji approved untuk evaluasi akhir
│       └── rejected/              # Data uji rejected untuk evaluasi akhir
└── processed/
    ├── training/
    │   └── approved-penerimaan/   # Data latih setelah resize (300x400)
    ├── validation/
    │   └── approved/              # Data validasi setelah resize (300x400)
    └── testing/
        ├── approved/              # Data uji approved setelah resize (300x400)
        └── rejected/              # Data uji rejected setelah resize (300x400)
```

---

## 3. Metodologi Pemisahan Data (Data Splitting)

Pemisahan data dilakukan secara ketat untuk mencegah kebocoran data (*data leakage*) dan bias optimistik:

Skrip `00_split_approved_dataset.py` memisahkan 20% data dari `dataset/raw/training/approved-penerimaan/` ke `dataset/raw/validation/approved/`.

- **Training Set**: Khusus digunakan untuk mengestimasi parameter model One-Class SVM dan *StandardScaler*.
- **Validation Set**: Digunakan untuk menetapkan nilai ambang batas (*threshold*) penerimaan pada persentil tertentu (misalnya persentil ke-5). Penggunaan *validation set* independen sangat krusial agar *decision threshold* tidak *overfit* terhadap data latih.
- **Testing Set**: Berisi data *approved* dan berbagai kategori *rejected* yang tidak pernah dilihat oleh model selama proses pelatihan maupun penentuan *threshold*. Digunakan murni untuk pelaporan metrik performa akhir (*unbiased evaluation*).

> **Catatan Metodologis:** Skrip pemisahan data menggunakan operasi pemindahan file (`shutil.move`). Jalankan skrip ini hanya sekali saat folder validasi masih kosong.

---

## 4. Instalasi dan Persiapan Lingkungan (Environment Setup)

Disarankan menggunakan *virtual environment* terisolasi untuk menghindari konflik dependensi.

### A. Membuat dan Mengaktifkan Virtual Environment

**macOS:**
```bash
python3 -m venv env-mac
source env-mac/bin/activate
```

**Linux:**
```bash
python3 -m venv env-linux
source env-linux/bin/activate
```

**Windows (Command Prompt):**
```bat
py -m venv env-windows
env-windows\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
py -m venv env-windows
env-windows\Scripts\Activate.ps1
```

### B. Mengaktifkan Kembali Virtual Environment yang Sudah Ada

- **macOS:** `source env-mac/bin/activate`
- **Linux:** `source env-linux/bin/activate`
- **Windows (CMD):** `env-windows\Scripts\activate.bat`
- **Windows (PowerShell):** `env-windows\Scripts\Activate.ps1`

### C. Menginstal Dependensi

Setelah *virtual environment* aktif, pasang pustaka yang diperlukan:

```bash
pip install -r requirements.txt
```

---

## 5. Konfigurasi Environment (`.env`)

Layanan API membaca konfigurasi dari berkas `.env`. Salin templat `.env.example` ke `.env`:

```bash
cp .env.example .env
```

Sesuaikan parameter berikut:
- `API_KEY`: Kunci rahasia untuk autentikasi request ke layanan backend.
- `API_KEY_HEADER_NAME`: Nama header HTTP untuk mengirim API Key (standar: `X-API-Key`).
- `ALLOWED_ORIGINS`: Daftar domain frontend yang diizinkan untuk CORS (pisahkan dengan koma).
- `APP_HOST`: Alamat IP host server (contoh: `0.0.0.0` atau `127.0.0.1`).
- `APP_PORT`: Port jaringan server (standar: `8000`).

---

## 6. Alur Eksekusi Pipeline (Pipeline Workflow)

Jalankan tahapan pemrosesan data, pelatihan, dan evaluasi secara berurutan:

```bash
# 1. Pisahkan 20% data approved untuk validasi independen
python 00_split_approved_dataset.py

# 2. Lakukan prapemrosesan dan standardisasi resolusi dataset ke 300x400 piksel RGB
python 01_resize_all_dataset.py

# 3. Ekstraksi vektor fitur visual 2048-D menggunakan ResNet50 pretrained
python 02_extract_features.py

# 4. Latih One-Class SVM dan hitung baseline threshold dari validation set
python 03_train_oneclass_svm.py

# 5. Uji inferensi pada satu sampel citra
python 06_predict_photo.py dataset/raw/testing/approved/contoh.jpg

# 6. Jalankan evaluasi menyeluruh pada seluruh testing set (approved & rejected)
python 04_evaluate_testing.py
```

### Inferensi Massal (Batch Prediction)
Untuk memproses seluruh foto dalam suatu direktori sekaligus:

```bash
python 06_predict_photo.py dataset/raw/testing/approved
```

Hasil inferensi massal akan diekspor ke:
- `results/batch_prediction_results.json`
- `results/batch_prediction_results.csv`

---

## 7. Layanan REST API (FastAPI Backend)

Implementasi backend siap produksi disediakan pada berkas `07_backend_api.py`.

### Menjalankan Server API

```bash
python 07_backend_api.py
```

Atau menggunakan ASGI server Uvicorn secara langsung:

```bash
uvicorn 07_backend_api:app --host 0.0.0.0 --port 8000
```

Dokumentasi interaktif Swagger UI dapat diakses melalui browser pada URL: `http://localhost:8000/docs`.

### Daftar Endpoint

| Metode | Endpoint | Deskripsi | Autentikasi |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Pemeriksaan status operasional layanan (*health check*) | Tidak |
| `POST` | `/predict` | Analisis kelayakan foto via upload `multipart/form-data` | Ya (`X-API-Key`) |
| `POST` | `/predict/base64` | Analisis kelayakan foto via JSON payload berformat Base64 | Ya (`X-API-Key`) |

### Contoh Pengujian Request

**1. Mengunggah Berkas Foto (`multipart/form-data`):**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "X-API-Key: rahasia-api-key-anda" \
  -F "photo=@dataset/raw/testing/approved/contoh.jpg"
```

**2. Mengirim Data Foto Base64 (JSON):**
```bash
curl -X POST "http://127.0.0.1:8000/predict/base64" \
  -H "X-API-Key: rahasia-api-key-anda" \
  -H "Content-Type: application/json" \
  -d '{"filename": "foto.jpg", "image_base64": "<string_base64>"}'
```

---

## 8. Evaluasi Model dan Eksperimen Tuning Threshold

1. **Evaluasi Testing (`04_evaluate_testing.py`)**:
   - Membaca model One-Class SVM dan ambang batas produksi dari `models/threshold.txt`.
   - Menguji performa terhadap data testing *approved* dan *rejected*.
   - Menyimpan metrik detail per foto ke `results/testing_evaluation.csv`.

2. **Eksperimen Tuning Threshold (`05_tune_threshold.py`)**:
   - Menganalisis kurva *trade-off* antara *False Acceptance Rate* (FAR) dan *False Rejection Rate* (FRR) dari hasil evaluasi testing.
   - Menghasilkan kandidat ambang batas optimal yang disimpan di `results/tuned_threshold.txt`.
   - Secara default, skrip ini **tidak menimpa** `models/threshold.txt` untuk menjaga validitas metodologis (menghindari *test-set snooping*).

Jika peneliti/pengembang secara sadar ingin menerapkan threshold hasil *tuning* tersebut ke model produksi:

```bash
python 05_tune_threshold.py --apply
python 04_evaluate_testing.py
```

---

## 9. Struktur Respon Prediksi (Output Schema)

- **`APPROVED`**: Foto memenuhi seluruh kriteria heuristik (rasio 4:3, tanpa watermark bawah, wajah tunggal, tidak buram, pencahayaan cukup, latar belakang seragam) serta memiliki skor kemiripan fitur visual di atas *threshold*.
- **`REJECTED`**: Foto melanggar satu atau lebih kriteria heuristik, atau skor kemiripan One-Class SVM berada di bawah *threshold*.
- **`recommendations`**: Rangkaian pesan umpan balik terstruktur yang menerangkan parameter apa saja yang tidak memenuhi syarat (misal: rasio dimensi tidak sesuai, pencahayaan terlalu gelap, atau terdeteksi teks/watermark).

---

## 10. Catatan Akademis dan Batasan Sistem (Future Work)

1. **Karakteristik One-Class Learning**: One-Class SVM mempelajari karakteristik intrinsik dari kelas normal (*approved*). Sistem tidak memodelkan kelas *rejected* secara eksplisit, melainkan mengidentifikasinya sebagai deviasi/pencilan (*outlier*).
2. **Peluang Riset Lanjutan**:
   - Apabila di masa mendatang data *rejected* telah terhimpun dalam kuantitas besar dan memiliki anotasi label yang seimbang per jenis kesalahan (misalnya: pose miring, latar belakang salah, pakaian non-formal), pendekatan dapat ditingkatkan menjadi *multi-class classification* atau *multi-task deep learning architecture*.
   - Eksplorasi ekstraktor fitur berbasis *Vision Transformer* (ViT) atau arsitektur *face recognition representation* (seperti ArcFace/InsightFace) untuk meningkatkan diskriminasi pose dan ekspresi wajah.
