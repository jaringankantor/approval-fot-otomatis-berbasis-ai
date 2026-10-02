from pathlib import Path

# Root project
BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Dataset asli / raw
# Folder ini diisi manual oleh Bapak.
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
# Dataset hasil resize / processed
# Folder ini diisi otomatis oleh script resize.
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
# Folder output
# ============================================================

MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# File model dan parameter
# ============================================================

SVM_MODEL_PATH = MODELS_DIR / "oneclass_svm.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.txt"


# ============================================================
# File fitur hasil ekstraksi ResNet50
# ============================================================

TRAINING_FEATURES_PATH = FEATURES_DIR / "training_features.npy"
VALIDATION_FEATURES_PATH = FEATURES_DIR / "validation_features.npy"
IMAGE_PATHS_PATH = FEATURES_DIR / "image_paths.txt"


# ============================================================
# File hasil evaluasi
# ============================================================

EVALUATION_RESULT_PATH = RESULTS_DIR / "testing_evaluation.csv"


# ============================================================
# Ukuran standar foto formal
# ============================================================

RESIZE_WIDTH = 300
RESIZE_HEIGHT = 400


# ============================================================
# Ekstensi gambar yang diterima
# ============================================================

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")