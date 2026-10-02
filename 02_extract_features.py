import numpy as np
from pathlib import Path

from src.config import (
    TRAIN_APPROVED_DIR,
    VALIDATION_APPROVED_DIR,
    FEATURES_DIR,
    TRAINING_FEATURES_PATH,
    VALIDATION_FEATURES_PATH,
    IMAGE_PATHS_PATH,
    IMAGE_EXTENSIONS
)

from src.feature_extractor import ResNet50FeatureExtractor


def get_image_paths(folder_path):
    """
    Mengambil semua path gambar dari folder tertentu.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder_path}")

    image_paths = []

    for image_path in folder_path.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(image_path)

    return sorted(image_paths)


def extract_features_from_paths(image_paths, extractor):
    """
    Melakukan ekstraksi fitur dari daftar path gambar.
    """
    features = []
    valid_paths = []

    for index, image_path in enumerate(image_paths, start=1):
        try:
            feature = extractor.extract(str(image_path))
            features.append(feature)
            valid_paths.append(str(image_path))

            print(f"[{index}/{len(image_paths)}] Berhasil ekstraksi: {image_path}")

        except Exception as error:
            print(f"[{index}/{len(image_paths)}] Gagal ekstraksi: {image_path}")
            print(f"Error: {error}")

    if len(features) == 0:
        raise ValueError("Tidak ada fitur yang berhasil diekstraksi.")

    return np.array(features), valid_paths


def save_image_paths(training_paths, validation_paths):
    """
    Menyimpan daftar path gambar yang berhasil diekstraksi.
    """
    with open(IMAGE_PATHS_PATH, "w", encoding="utf-8") as file:
        file.write("[TRAINING APPROVED]\n")
        for path in training_paths:
            file.write(path + "\n")

        file.write("\n[VALIDATION APPROVED]\n")
        for path in validation_paths:
            file.write(path + "\n")


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Memulai ekstraksi fitur menggunakan ResNet50 pretrained...")
    print(f"Folder training   : {TRAIN_APPROVED_DIR}")
    print(f"Folder validation : {VALIDATION_APPROVED_DIR}")

    extractor = ResNet50FeatureExtractor()

    training_image_paths = get_image_paths(TRAIN_APPROVED_DIR)
    validation_image_paths = get_image_paths(VALIDATION_APPROVED_DIR)

    print(f"\nJumlah gambar training approved   : {len(training_image_paths)}")
    print(f"Jumlah gambar validation approved : {len(validation_image_paths)}")

    if len(training_image_paths) == 0:
        raise ValueError("Folder training approved kosong.")

    if len(validation_image_paths) == 0:
        raise ValueError("Folder validation approved kosong.")

    print("\nEkstraksi fitur training approved...")
    training_features, valid_training_paths = extract_features_from_paths(
        training_image_paths,
        extractor
    )

    print("\nEkstraksi fitur validation approved...")
    validation_features, valid_validation_paths = extract_features_from_paths(
        validation_image_paths,
        extractor
    )

    np.save(TRAINING_FEATURES_PATH, training_features)
    np.save(VALIDATION_FEATURES_PATH, validation_features)

    save_image_paths(valid_training_paths, valid_validation_paths)

    print("\nEkstraksi fitur selesai.")
    print(f"Training features disimpan di   : {TRAINING_FEATURES_PATH}")
    print(f"Validation features disimpan di : {VALIDATION_FEATURES_PATH}")
    print(f"Daftar path gambar disimpan di  : {IMAGE_PATHS_PATH}")

    print("\nUkuran fitur:")
    print(f"Training   : {training_features.shape}")
    print(f"Validation : {validation_features.shape}")


if __name__ == "__main__":
    main()
