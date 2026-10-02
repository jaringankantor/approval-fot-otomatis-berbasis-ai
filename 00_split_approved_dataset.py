import random
import shutil

from src.config import (
    RAW_TRAIN_APPROVED_DIR,
    RAW_VALIDATION_APPROVED_DIR,
    IMAGE_EXTENSIONS
)


# Direktori sumber data approved dan direktori target validasi
SOURCE_DIR = RAW_TRAIN_APPROVED_DIR
VALIDATION_DIR = RAW_VALIDATION_APPROVED_DIR

# Proporsi data validasi (20%) dan seed untuk pengacakan yang konsisten
VALIDATION_RATIO = 0.2
RANDOM_SEED = 42


def get_images(folder):
    """
    Mengambil daftar path file gambar yang valid dari folder yang ditentukan.
    """
    return [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():
    """
    Memisahkan sebagian data approved dari folder training ke folder validation
    sesuai rasio yang ditentukan, jika folder validation masih kosong.
    """
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {SOURCE_DIR}")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    existing_validation_images = get_images(VALIDATION_DIR)

    if len(existing_validation_images) > 0:
        print(f"Folder validation sudah berisi {len(existing_validation_images)} gambar.")
        print("Split dibatalkan agar data validation tidak tercampur atau bertambah ganda.")
        return

    images = get_images(SOURCE_DIR)

    if len(images) == 0:
        raise ValueError(f"Tidak ada gambar di folder: {SOURCE_DIR}")

    random.seed(RANDOM_SEED)
    random.shuffle(images)

    validation_count = int(len(images) * VALIDATION_RATIO)
    validation_images = images[:validation_count]

    print(f"Total gambar approved : {len(images)}")
    print(f"Jumlah validation     : {validation_count}")
    print(f"Jumlah training       : {len(images) - validation_count}")

    for image_path in validation_images:
        destination = VALIDATION_DIR / image_path.name
        shutil.move(str(image_path), str(destination))
        print(f"Pindah ke validation: {image_path.name}")

    print("\nSelesai split dataset approved.")


if __name__ == "__main__":
    main()
