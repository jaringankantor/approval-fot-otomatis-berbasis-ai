from pathlib import Path

# Direktori utama proyek
BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Direktori dataset asli (raw)
# ============================================================

RAW_DATASET_DIR = BASE_DIR / "dataset" / "raw"

RAW_TRAIN_DIR = RAW_DATASET_DIR / "training"
RAW_VALIDATION_DIR = RAW_DATASET_DIR / "validation"
RAW_TESTING_DIR = RAW_DATASET_DIR / "testing"

RAW_TRAIN_APPROVED_DIR = RAW_TRAIN_DIR / "approved-penerimaan"
RAW_VALIDATION_APPROVED_DIR = RAW_VALIDATION_DIR / "approved"

RAW_TESTING_APPROVED_DIR = RAW_TESTING_DIR / "approved"
RAW_TESTING_REJECTED_DIR = RAW_TESTING_DIR / "rejected"


# ============================================================
# Direktori dataset hasil prapemrosesan (processed)
# ============================================================

PROCESSED_DATASET_DIR = BASE_DIR / "dataset" / "processed"

TRAIN_DIR = PROCESSED_DATASET_DIR / "training"
VALIDATION_DIR = PROCESSED_DATASET_DIR / "validation"
TESTING_DIR = PROCESSED_DATASET_DIR / "testing"

TRAIN_APPROVED_DIR = TRAIN_DIR / "approved-penerimaan"
VALIDATION_APPROVED_DIR = VALIDATION_DIR / "approved"

TESTING_APPROVED_DIR = TESTING_DIR / "approved"
TESTING_REJECTED_DIR = TESTING_DIR / "rejected"


# ============================================================
# Direktori penyimpanan output
# ============================================================

MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# Path penyimpanan model dan parameter ambang batas
# ============================================================

SVM_MODEL_PATH = MODELS_DIR / "oneclass_svm.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.txt"


# ============================================================
# Path penyimpanan fitur hasil ekstraksi ResNet50
# ============================================================

TRAINING_FEATURES_PATH = FEATURES_DIR / "training_features.npy"
VALIDATION_FEATURES_PATH = FEATURES_DIR / "validation_features.npy"
IMAGE_PATHS_PATH = FEATURES_DIR / "image_paths.txt"


# ============================================================
# Path penyimpanan laporan hasil evaluasi
# ============================================================

EVALUATION_RESULT_PATH = RESULTS_DIR / "testing_evaluation.csv"


# ============================================================
# Dimensi standar pas foto formal (lebar x tinggi)
# ============================================================

RESIZE_WIDTH = 300
RESIZE_HEIGHT = 400


# ============================================================
# Format ekstensi gambar yang didukung
# ============================================================

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")