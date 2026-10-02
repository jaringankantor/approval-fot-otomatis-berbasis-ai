# Approval Foto Otomatis Berbasis AI

Project ini menggunakan:

- ResNet50 pretrained sebagai ekstraktor fitur foto approved.
- One-Class SVM untuk mempelajari pola foto approved saja.
- Heuristik tambahan untuk rekomendasi blur, pencahayaan, dan background.
- Folder rejected digunakan untuk testing/evaluasi, bukan untuk melatih model utama.

## Struktur dataset

```text
dataset/
├── raw/
│   ├── training/
│   │   └── approved-penerimaan/
│   ├── validation/
│   │   └── approved/
│   └── testing/
│       ├── approved/
│       └── rejected/
└── processed/
    ├── training/
    │   └── approved-penerimaan/
    ├── validation/
    │   └── approved/
    └── testing/
        ├── approved/
        └── rejected/
```

## Split training dan validation

Script `00_split_approved_dataset.py` digunakan untuk mengambil sebagian foto approved dari:

```text
dataset/raw/training/approved-penerimaan
```

lalu memindahkannya ke:

```text
dataset/raw/validation/approved
```

Pemisahan ini penting karena `training` dan `validation` memiliki fungsi berbeda:

- `training`: dipakai untuk melatih One-Class SVM agar mengenali pola foto approved.
- `validation`: dipakai untuk menentukan threshold penerimaan dari foto approved yang tidak ikut dilatih.
- `testing`: dipakai untuk evaluasi akhir setelah model dan threshold selesai dibuat.

Jika training dan validation memakai foto yang sama, threshold bisa terlalu optimistis karena model diuji pada data yang sudah pernah dipelajari. Akibatnya hasil terlihat bagus di data lama, tetapi lebih mudah salah saat menerima foto baru.

Jalankan script split hanya jika folder `dataset/raw/validation/approved` masih kosong atau belum dibuat. Script ini memakai `shutil.move`, sehingga file benar-benar dipindahkan dari training ke validation. Jika folder validation sudah berisi gambar, proses akan dibatalkan agar data validation tidak bertambah ganda.

## Instalasi

Buat virtual environment terlebih dahulu dari root folder project sesuai OS yang digunakan.

macOS:

```bash
python3 -m venv env-mac
source env-mac/bin/activate
```

Linux:

```bash
python3 -m venv env-linux
source env-linux/bin/activate
```

Windows Command Prompt:

```bat
py -m venv env-windows
env-windows\Scripts\activate.bat
```

Windows PowerShell:

```powershell
py -m venv env-windows
env-windows\Scripts\Activate.ps1
```

Jika virtual environment sudah pernah dibuat, cukup aktifkan kembali.

macOS:

```bash
source env-mac/bin/activate
```

Linux:

```bash
source env-linux/bin/activate
```

Windows Command Prompt:

```bat
env-windows\Scripts\activate.bat
```

Windows PowerShell:

```powershell
env-windows\Scripts\Activate.ps1
```

Setelah virtual environment aktif, install dependency:

```bash
pip install -r requirements.txt
```

Catatan: `joblib` biasanya ikut terpasang bersama scikit-learn, tetapi aman jika dipasang eksplisit.

## Konfigurasi environment

Backend membaca konfigurasi dari file `.env`. File ini berisi data sensitif dan tidak boleh dibagikan.

Contoh konfigurasi tersedia di `.env.example`.

```bash
cp .env.example .env
```

Isi utama:

- `API_KEY`: kunci rahasia untuk aplikasi internal yang akan mengakses backend.
- `API_KEY_HEADER_NAME`: nama header HTTP untuk mengirim API key, default `X-API-Key`.
- `ALLOWED_ORIGINS`: daftar origin frontend yang diizinkan untuk CORS, pisahkan dengan koma.
- `APP_HOST`: host server FastAPI.
- `APP_PORT`: port server FastAPI.

## Cara menjalankan

Dari root folder project, setelah virtual environment aktif:

```bash
python 00_split_approved_dataset.py
python 01_resize_all_dataset.py
python 02_extract_features.py
python 03_train_oneclass_svm.py
python 06_predict_photo.py dataset/raw/testing/approved/contoh.jpg
python 04_evaluate_testing.py
```

Prediksi satu folder foto:

```bash
python 06_predict_photo.py dataset/processed/testing/approved
```

Hasil prediksi massal akan disimpan ke:

- `results/batch_prediction_results.json`
- `results/batch_prediction_results.csv`

## Menjalankan backend FastAPI

Backend API tersedia di `07_backend_api.py`.

Jalankan server setelah virtual environment aktif:

```bash
python 07_backend_api.py
```

Atau langsung dengan Uvicorn:

```bash
uvicorn 07_backend_api:app --host 0.0.0.0 --port 8000
```

Endpoint:

- `GET /health`: cek status server.
- `POST /predict`: upload foto dengan `multipart/form-data`, field `photo`, wajib memakai header API key.
- `POST /predict/base64`: kirim foto dalam JSON base64, wajib memakai header API key.
- `GET /docs`: dokumentasi Swagger otomatis dari FastAPI.

Contoh upload foto:

```bash
curl -H "X-API-Key: isi-api-key-dari-env" \
  -F photo=@dataset/raw/testing/approved/f_foto-944210.jpg \
  http://127.0.0.1:8000/predict
```

Contoh request JSON base64:

```bash
curl -H "X-API-Key: isi-api-key-dari-env" \
  -H "Content-Type: application/json" \
  -d '{"filename":"foto.jpg","image_base64":"base64-foto"}' \
  http://127.0.0.1:8000/predict/base64
```

Opsional setelah melihat hasil evaluasi:

```bash
python 05_tune_threshold.py
```

`04_evaluate_testing.py` aman dijalankan untuk mengukur performa pada data testing. Script ini hanya membaca model dan menyimpan hasil evaluasi ke `results/testing_evaluation.csv`.

`05_tune_threshold.py` hanya digunakan untuk eksperimen mencari kandidat threshold dari hasil evaluasi testing. Secara default script ini tidak mengubah `models/threshold.txt`, tetapi menyimpan kandidat threshold ke `results/tuned_threshold.txt`.

Jangan langsung memakai threshold dari testing sebagai threshold produksi, karena testing sebaiknya tetap menjadi data evaluasi akhir. Threshold produksi utama dibuat oleh `03_train_oneclass_svm.py` dari data validation. Jika benar-benar ingin menerapkan hasil tuning testing, jalankan:

```bash
python 05_tune_threshold.py --apply
python 04_evaluate_testing.py
```

## Interpretasi hasil prediksi

- `APPROVED`: foto mirip pola approved dan tidak melanggar aturan sederhana blur/pencahayaan/background.
- `REJECTED`: foto dianggap tidak sesuai.
- `recommendations`: alasan sederhana yang bisa ditampilkan ke pengguna.

## Catatan penting

One-Class SVM tidak belajar kategori rejected secara spesifik. Jika ingin klasifikasi penyebab reject secara lebih akurat, kumpulkan data rejected yang cukup per kategori lalu lanjutkan ke model supervised multi-class.
