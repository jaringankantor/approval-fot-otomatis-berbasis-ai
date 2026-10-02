import base64
import tempfile
import threading
from pathlib import Path

import numpy as np

from src.config import IMAGE_EXTENSIONS
from src.predictor import PhotoApprovalPredictor


MAX_UPLOAD_BYTES = 10 * 1024 * 1024

_predictor = None
_predictor_lock = threading.Lock()


def get_predictor():
    """
    Mengambil instance singleton dari PhotoApprovalPredictor dengan thread-safety lock.
    """
    global _predictor

    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = PhotoApprovalPredictor()

    return _predictor


def make_json_safe(value):
    """
    Mengonversi tipe data numpy (int, float) dan Path menjadi tipe data primitif Python
    agar dapat diserialisasi ke dalam format JSON tanpa error.
    """
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, Path):
        return str(value)

    return value


def validate_filename(filename):
    """
    Memvalidasi ekstensi nama file terhadap daftar format gambar yang didukung.
    """
    if not filename:
        return "uploaded.jpg"

    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        supported_formats = ", ".join(IMAGE_EXTENSIONS)
        raise ValueError(f"Format file tidak didukung. Gunakan: {supported_formats}")

    return filename


def validate_file_size(image_bytes):
    """
    Memastikan data byte gambar tidak kosong dan tidak melebihi batas ukuran maksimal (10 MB).
    """
    if not image_bytes:
        raise ValueError("File foto kosong.")

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Ukuran upload maksimal 10 MB.")


def decode_base64_photo(image_base64):
    """
    Mendekode string base64 menjadi byte citra digital, mendukung Data URL prefix (data:image/...).
    """
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        return base64.b64decode(image_base64, validate=True)
    except ValueError as exc:
        raise ValueError("Field `image_base64` bukan base64 yang valid.") from exc


def write_temp_image(image_bytes, filename):
    """
    Menulis byte gambar ke file sementara di disk untuk diproses oleh prediktor.
    """
    filename = validate_filename(filename)
    validate_file_size(image_bytes)

    suffix = Path(filename).suffix.lower()
    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        prefix="photo_check_",
        delete=False
    )

    try:
        temp_file.write(image_bytes)
        return Path(temp_file.name)
    finally:
        temp_file.close()


def predict_image_bytes(image_bytes, filename):
    """
    Menerima byte gambar, menyimpannya sementara, menjalankan prediksi kelayakan,
    serta membersihkan file sementara setelah selesai.
    """
    temp_path = None

    try:
        temp_path = write_temp_image(image_bytes, filename)
        result = get_predictor().predict(temp_path)
        result["uploaded_filename"] = filename
        return make_json_safe(result)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
