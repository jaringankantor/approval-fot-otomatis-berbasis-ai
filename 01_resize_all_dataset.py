import os
import stat
from PIL import Image, ImageFile

from src.config import (
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    RESIZE_WIDTH,
    RESIZE_HEIGHT,
    IMAGE_EXTENSIONS
)

# Agar PIL lebih toleran terhadap gambar yang sedikit truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def remove_file(path):
    try:
        path.unlink()
    except PermissionError:
        os.chmod(path.parent, stat.S_IWRITE)
        os.chmod(path, stat.S_IWRITE)

        try:
            path.unlink()
        except PermissionError as error:
            raise PermissionError(
                f"File tidak bisa dihapus: {path}. "
                "Tutup file jika sedang dibuka atau tunggu sinkronisasi Google Drive selesai."
            ) from error


def remove_empty_directory(path):
    try:
        os.chmod(path, stat.S_IWRITE)
        path.rmdir()
    except OSError:
        # Folder kosong yang read-only/terkunci boleh ditinggalkan.
        # File di dalamnya sudah dihapus oleh remove_file.
        pass


def clear_processed_dataset():
    """
    Menghapus isi folder output processed sebelum resize ulang.
    Folder utamanya tetap dipertahankan.
    """
    if not PROCESSED_DATASET_DIR.exists():
        PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        return

    if PROCESSED_DATASET_DIR.resolve() == RAW_DATASET_DIR.resolve():
        raise ValueError("Folder output processed tidak boleh sama dengan folder raw.")

    items = sorted(
        PROCESSED_DATASET_DIR.rglob("*"),
        key=lambda path: len(path.parts),
        reverse=True
    )

    for item in items:
        if item.is_file() or item.is_symlink():
            remove_file(item)

    for item in items:
        if item.is_dir():
            remove_empty_directory(item)


def resize_image(input_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        image = Image.open(input_path)
        image.load()
        image = image.convert("RGB")
        image = image.resize((RESIZE_WIDTH, RESIZE_HEIGHT))
        image.save(output_path)

        return True, ""

    except Exception as error:
        return False, str(error)


def resize_all_dataset():
    if not RAW_DATASET_DIR.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {RAW_DATASET_DIR}")

    clear_processed_dataset()
    print(f"Folder output dikosongkan: {PROCESSED_DATASET_DIR}")

    total_success = 0
    total_failed = 0
    failed_files = []

    for image_path in RAW_DATASET_DIR.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            relative_path = image_path.relative_to(RAW_DATASET_DIR)
            output_path = PROCESSED_DATASET_DIR / relative_path

            success, error_message = resize_image(image_path, output_path)

            if success:
                total_success += 1
                print(f"Resize OK: {image_path} -> {output_path}")
            else:
                total_failed += 1
                failed_files.append((image_path, error_message))
                print(f"Resize GAGAL: {image_path}")
                print(f"Error       : {error_message}")

    print("\nSelesai resize.")
    print(f"Total berhasil: {total_success}")
    print(f"Total gagal   : {total_failed}")
    print(f"Hasil disimpan di: {PROCESSED_DATASET_DIR}")

    if total_failed > 0:
        log_path = PROCESSED_DATASET_DIR / "resize_failed_files.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w", encoding="utf-8") as file:
            for path, error in failed_files:
                file.write(f"{path} | {error}\n")

        print(f"\nDaftar file gagal disimpan di: {log_path}")
        print("File yang gagal sebaiknya dicek, diunduh ulang, atau dihapus dari dataset raw.")


if __name__ == "__main__":
    resize_all_dataset()
