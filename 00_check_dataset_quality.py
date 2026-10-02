import cv2
import numpy as np

from src.config import (
    RAW_TRAIN_APPROVED_DIR,
    IMAGE_EXTENSIONS,
    RESIZE_WIDTH,
    RESIZE_HEIGHT
)


# ============================================================
# Konfigurasi rasio foto potret 4:3
# Standar dimensi: lebar 300 piksel, tinggi 400 piksel
# Target rasio: lebar / tinggi = 300 / 400 = 0.75
# ============================================================

TARGET_RATIO = RESIZE_WIDTH / RESIZE_HEIGHT
RATIO_TOLERANCE = 0.10

MIN_RATIO = TARGET_RATIO * (1 - RATIO_TOLERANCE)
MAX_RATIO = TARGET_RATIO * (1 + RATIO_TOLERANCE)


# ============================================================
# Ambang batas awal penentuan kualitas foto
# Nilai referensi untuk ketajaman (blur) dan pencahayaan (brightness)
# ============================================================

MIN_ACCEPTABLE_BLUR = 100
MIN_ACCEPTABLE_BRIGHTNESS = 90
MAX_ACCEPTABLE_BRIGHTNESS = 180


def calculate_blur_brightness_ratio(image_path):
    """
    Menghitung metrik kualitas foto:
    - Tingkat ketajaman (variansi Laplacian)
    - Rata-rata intensitas pencahayaan (brightness)
    - Rasio aspek foto (lebar / tinggi)
    """
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_score = np.mean(gray)
    ratio = width / height

    return {
        "file": image_path,
        "filename": image_path.name,
        "width": width,
        "height": height,
        "ratio": ratio,
        "blur": blur_score,
        "brightness": brightness_score
    }


def print_filename_list(title, items):
    """
    Menampilkan daftar nama file berdasarkan kategori tertentu.
    """
    print(f"\n--- {title} ---")

    if not items:
        print("Tidak ada.")
        return

    filenames = [item["filename"] for item in items]
    print(", ".join(filenames))


def main():
    """
    Menjalankan proses analisis kualitas gambar pada folder dataset approved.
    Mengevaluasi distribusi blur, brightness, dan rasio aspek.
    """
    dataset_dir = RAW_TRAIN_APPROVED_DIR

    results = []
    failed_files = []

    image_files = [
        file for file in dataset_dir.rglob("*")
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"Tidak ada file gambar ditemukan di folder: {dataset_dir}")
        return

    print("=== ANALISIS DATASET APPROVED ===")
    print(f"Folder dataset : {dataset_dir}")
    print(f"Jumlah foto ditemukan: {len(image_files)}")
    print("Memproses foto...\n")

    for image_path in image_files:
        result = calculate_blur_brightness_ratio(image_path)

        if result is None:
            failed_files.append(image_path)
            continue

        results.append(result)

    if not results:
        print("Tidak ada foto yang berhasil diproses.")
        return

    blur_values = np.array([item["blur"] for item in results])
    brightness_values = np.array([item["brightness"] for item in results])
    ratio_values = np.array([item["ratio"] for item in results])

    invalid_ratio_files = [
        item for item in results
        if item["ratio"] < MIN_RATIO or item["ratio"] > MAX_RATIO
    ]

    invalid_blur_files = [
        item for item in results
        if item["blur"] < MIN_ACCEPTABLE_BLUR
    ]

    invalid_brightness_files = [
        item for item in results
        if (
            item["brightness"] < MIN_ACCEPTABLE_BRIGHTNESS
            or item["brightness"] > MAX_ACCEPTABLE_BRIGHTNESS
        )
    ]

    print("=== HASIL ANALISIS DATASET APPROVED ===")
    print(f"Jumlah foto berhasil diproses : {len(results)}")
    print(f"Jumlah foto gagal dibaca      : {len(failed_files)}")

    print("\n--- NILAI BLUR ---")
    print(f"Minimum   : {blur_values.min():.2f}")
    print(f"Maksimum  : {blur_values.max():.2f}")
    print(f"Rata-rata : {blur_values.mean():.2f}")

    print("\n--- NILAI BRIGHTNESS ---")
    print(f"Minimum   : {brightness_values.min():.2f}")
    print(f"Maksimum  : {brightness_values.max():.2f}")
    print(f"Rata-rata : {brightness_values.mean():.2f}")

    print("\n--- RASIO FOTO WIDTH/HEIGHT ---")
    print(f"Ukuran standar foto           : {RESIZE_WIDTH}x{RESIZE_HEIGHT}")
    print(f"Target rasio portrait 4:3     : {TARGET_RATIO:.4f}")
    print(f"Toleransi rasio               : {RATIO_TOLERANCE * 100:.0f}%")
    print(f"Batas rasio minimum           : {MIN_RATIO:.4f}")
    print(f"Batas rasio maksimum          : {MAX_RATIO:.4f}")
    print(f"Minimum rasio dataset         : {ratio_values.min():.4f}")
    print(f"Maksimum rasio dataset        : {ratio_values.max():.4f}")
    print(f"Rata-rata rasio dataset       : {ratio_values.mean():.4f}")

    print("\n--- DAFTAR SEMUA FOTO DAN RASIO ---")
    for item in results:
        print(
            f"{item['filename']} | "
            f"{item['width']}x{item['height']} | "
            f"rasio={item['ratio']:.4f}"
        )

    print_filename_list(
        "FOTO DENGAN RASIO TIDAK SESUAI PORTRAIT 4:3",
        invalid_ratio_files
    )

    print_filename_list(
        "FOTO DENGAN NILAI BLUR TIDAK COCOK UNTUK DATASET",
        invalid_blur_files
    )

    print_filename_list(
        "FOTO DENGAN NILAI BRIGHTNESS TIDAK COCOK UNTUK DATASET",
        invalid_brightness_files
    )

    print("\n--- DETAIL FOTO DENGAN RASIO TIDAK SESUAI PORTRAIT 4:3 ---")
    if invalid_ratio_files:
        for item in invalid_ratio_files:
            print(
                f"{item['filename']} | "
                f"{item['width']}x{item['height']} | "
                f"rasio={item['ratio']:.4f}"
            )
    else:
        print("Tidak ada.")

    print("\n--- DETAIL FOTO DENGAN NILAI BLUR TIDAK COCOK UNTUK DATASET ---")
    if invalid_blur_files:
        for item in sorted(invalid_blur_files, key=lambda x: x["blur"]):
            print(
                f"{item['filename']} | "
                f"blur={item['blur']:.2f} | "
                f"brightness={item['brightness']:.2f} | "
                f"{item['width']}x{item['height']} | "
                f"rasio={item['ratio']:.4f}"
            )
    else:
        print("Tidak ada.")

    print("\n--- DETAIL FOTO DENGAN NILAI BRIGHTNESS TIDAK COCOK UNTUK DATASET ---")
    if invalid_brightness_files:
        for item in sorted(invalid_brightness_files, key=lambda x: x["brightness"]):
            print(
                f"{item['filename']} | "
                f"brightness={item['brightness']:.2f} | "
                f"blur={item['blur']:.2f} | "
                f"{item['width']}x{item['height']} | "
                f"rasio={item['ratio']:.4f}"
            )
    else:
        print("Tidak ada.")

    if failed_files:
        print("\n--- FILE GAGAL DIBACA ---")
        print(", ".join([file.name for file in failed_files]))


if __name__ == "__main__":
    main()