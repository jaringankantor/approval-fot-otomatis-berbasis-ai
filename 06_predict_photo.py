import sys
import csv
import json
from pathlib import Path

from src.config import IMAGE_EXTENSIONS, RESULTS_DIR
from src.predictor import PhotoApprovalPredictor


def print_result(result):
    """
    Menampilkan ringkasan hasil prediksi ke terminal.
    Untuk foto APPROVED, hanya informasi status dan kesesuaian pola yang ditampilkan.
    Untuk foto REJECTED, detail penyebab penolakan dan rekomendasi perbaikan akan ditampilkan.
    """
    print("\n" + "=" * 70)
    print("HASIL PREDIKSI FOTO")
    print("=" * 70)

    print(f"File Foto          : {result['image_path']}")
    print(f"Status             : {result['status']}")
    print(f"AI Status          : {result.get('ai_status', '-')}")
    print(f"Rejection Source   : {result.get('rejection_source', '-')}")
    print(f"AI Similarity Score: {result['ai_similarity_score']:.6f}")
    print(f"Threshold          : {result['threshold']:.6f}")

    if result["status"] == "APPROVED":
        print(f"Pesan              : {result.get('message', 'Foto diterima oleh AI.')}")

        print("\nRekomendasi:")
        for index, recommendation in enumerate(result.get("recommendations", []), start=1):
            print(f"{index}. {recommendation}")

        print("=" * 70)
        return

    print("\nANALISIS TAMBAHAN UNTUK FOTO REJECTED")
    print("-" * 70)

    print(f"Watermark Status   : {result.get('watermark_status', '-')}")
    print(f"Ratio Status       : {result.get('ratio_status', '-')}")
    print(f"Photo Width        : {result.get('photo_width', '-')}")
    print(f"Photo Height       : {result.get('photo_height', '-')}")
    print(f"Photo Ratio        : {result.get('photo_ratio', '-')}")
    print(f"Face Status        : {result.get('face_status', '-')}")
    print(f"Face Count         : {result.get('face_count', '-')}")

    blur_score = result.get("blur_score", "")
    if blur_score != "":
        print(f"Blur Score         : {float(blur_score):.2f}")
    else:
        print("Blur Score         : -")

    print(f"Blur Status        : {result.get('blur_status', '-')}")

    brightness_score = result.get("brightness_score", "")
    if brightness_score != "":
        print(f"Brightness Score   : {float(brightness_score):.2f}")
    else:
        print("Brightness Score   : -")

    print(f"Brightness Status  : {result.get('brightness_status', '-')}")
    print(f"Background Status  : {result.get('background_status', '-')}")

    print("\nKeterangan/Rekomendasi:")
    for index, recommendation in enumerate(result.get("recommendations", []), start=1):
        print(f"{index}. {recommendation}")

    print("=" * 70)


def save_result_json(result, output_path):
    """
    Menyimpan hasil prediksi ke file JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    print(f"\nHasil JSON disimpan di: {output_path}")


def save_results_csv(results, output_path):
    """
    Menyimpan ringkasan hasil prediksi massal ke file CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
        "status",
        "ai_status",
        "rejection_source",
        "ai_similarity_score",
        "threshold",
        "message",
        "recommendations"
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {}
            for fieldname in fieldnames:
                value = result.get(fieldname, "")
                if fieldname == "recommendations" and isinstance(value, list):
                    value = " | ".join(value)
                row[fieldname] = value

            writer.writerow(row)

    print(f"Hasil CSV disimpan di : {output_path}")


def validate_input_path(input_path):
    """
    Mengecek file/folder input sebelum diprediksi.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Path tidak ditemukan: {input_path}")

    if input_path.is_file() and input_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Format file tidak didukung: {input_path.suffix}. "
            f"Gunakan format: {IMAGE_EXTENSIONS}"
        )

    return input_path


def collect_image_files(input_dir):
    """
    Mengambil semua file gambar dari folder secara rekursif.
    """
    input_dir = Path(input_dir)
    image_files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_files)


def predict_single_file(predictor, input_photo_path):
    """
    Menjalankan inferensi pada satu file foto, mencetak hasil ke terminal,
    dan menyimpannya ke file JSON.
    """
    # Inferensi dilakukan langsung menggunakan file foto asli tanpa resize awal
    # agar validasi rasio aspek foto tetap akurat sesuai dimensi aslinya.
    result = predictor.predict(input_photo_path)

    print_result(result)

    save_result_json(
        result,
        output_path=RESULTS_DIR / "single_prediction_result.json"
    )


def predict_folder(predictor, input_dir):
    """
    Menjalankan inferensi secara massal (batch) pada seluruh foto di dalam folder,
    mencetak ringkasan, serta mengekspor hasil ke format JSON dan CSV.
    """
    image_files = collect_image_files(input_dir)

    if len(image_files) == 0:
        raise ValueError(
            f"Tidak ada file gambar di folder: {input_dir}. "
            f"Gunakan format: {IMAGE_EXTENSIONS}"
        )

    print(f"Jumlah foto ditemukan: {len(image_files)}")

    results = []
    for index, image_path in enumerate(image_files, start=1):
        print(f"\n[{index}/{len(image_files)}] Memproses: {image_path}")
        result = predictor.predict(image_path)
        results.append(result)
        print(
            f"Status: {result['status']} | "
            f"AI Score: {result['ai_similarity_score']:.6f} | "
            f"Source: {result.get('rejection_source', '-') or '-'}"
        )

    approved_count = sum(1 for result in results if result["status"] == "APPROVED")
    rejected_count = sum(1 for result in results if result["status"] == "REJECTED")

    print("\n" + "=" * 70)
    print("RINGKASAN PREDIKSI MASSAL")
    print("=" * 70)
    print(f"Total Foto : {len(results)}")
    print(f"APPROVED   : {approved_count}")
    print(f"REJECTED   : {rejected_count}")
    print("=" * 70)

    output_json_path = RESULTS_DIR / "batch_prediction_results.json"
    output_csv_path = RESULTS_DIR / "batch_prediction_results.csv"

    save_result_json(results, output_json_path)
    save_results_csv(results, output_csv_path)


def main():
    """
    Titik masuk utama CLI untuk memvalidasi argumen masukan (path file/folder)
    dan mengarahkan eksekusi ke mode single-file atau batch-folder.
    """
    if len(sys.argv) < 2:
        print("Gunakan perintah:")
        print("python 06_predict_photo.py path/ke/foto.jpg")
        print("python 06_predict_photo.py path/ke/folder_foto")
        print("\nContoh:")
        print("python 06_predict_photo.py dataset/raw/testing/approved/contoh.jpg")
        print("python 06_predict_photo.py dataset/raw/testing/approved")
        sys.exit(1)

    input_path = validate_input_path(sys.argv[1])

    print("Memproses prediksi foto...")
    print(f"Input: {input_path}")

    predictor = PhotoApprovalPredictor()

    if input_path.is_dir():
        predict_folder(predictor, input_path)
        return

    predict_single_file(predictor, input_path)


if __name__ == "__main__":
    main()
