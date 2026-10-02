import csv
from pathlib import Path

from src.config import (
    TESTING_APPROVED_DIR,
    TESTING_REJECTED_DIR,
    RESULTS_DIR,
    EVALUATION_RESULT_PATH,
    IMAGE_EXTENSIONS
)

from src.predictor import PhotoApprovalPredictor


TESTING_CLASSES = [
    {
        "folder": TESTING_APPROVED_DIR,
        "label": "approved",
        "expected_status": "APPROVED"
    },
    {
        "folder": TESTING_REJECTED_DIR,
        "label": "rejected",
        "expected_status": "REJECTED"
    }
]


def get_image_paths(folder_path):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"Folder tidak ditemukan, dilewati: {folder_path}")
        return []

    image_paths = []

    for image_path in folder_path.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(image_path)

    return sorted(image_paths)


def evaluate():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictor = PhotoApprovalPredictor()

    rows = []
    total = 0
    correct = 0

    print("Memulai evaluasi dataset testing...\n")

    for test_class in TESTING_CLASSES:
        folder = test_class["folder"]
        label = test_class["label"]
        expected_status = test_class["expected_status"]

        image_paths = get_image_paths(folder)

        print(f"Folder          : {folder}")
        print(f"Label           : {label}")
        print(f"Expected Status : {expected_status}")
        print(f"Jumlah gambar   : {len(image_paths)}\n")

        for image_path in image_paths:
            result = predictor.predict(str(image_path))

            predicted_status = result["status"]
            is_correct = predicted_status == expected_status

            total += 1

            if is_correct:
                correct += 1

            rows.append({
                "image_path": str(image_path),
                "file_name": image_path.name,
                "true_label": label,
                "expected_status": expected_status,
                "predicted_status": predicted_status,
                "is_correct": is_correct,
                "ai_similarity_score": result.get("ai_similarity_score"),
                "threshold": result.get("threshold"),
                "message": result.get("message", ""),

                # Kolom-kolom di bawah ini hanya terisi jika AI memutuskan REJECTED.
                # Jika APPROVED, nilainya akan kosong.
                "face_status": result.get("face_status", ""),
                "face_count": result.get("face_count", ""),
                "blur_score": result.get("blur_score", ""),
                "blur_status": result.get("blur_status", ""),
                "brightness_score": result.get("brightness_score", ""),
                "brightness_status": result.get("brightness_status", ""),
                "background_status": result.get("background_status", ""),
                "background_avg_std": result.get("background_avg_std", ""),
                "background_mean_distance": result.get("background_mean_distance", ""),
                "background_outlier_ratio": result.get("background_outlier_ratio", ""),
                "background_brightness": result.get("background_brightness", ""),
                "recommendations": " | ".join(result.get("recommendations", [])),

                "watermark_status": result.get("watermark_status", ""),
                "watermark_score": result.get("watermark_score", ""),
                "watermark_dark_ratio": result.get("watermark_dark_ratio", ""),
                "watermark_edge_ratio": result.get("watermark_edge_ratio", ""),
                "watermark_white_ratio": result.get("watermark_white_ratio", ""),
                "ratio_status": result.get("ratio_status", ""),
                "photo_width": result.get("photo_width", ""),
                "photo_height": result.get("photo_height", ""),
                "photo_ratio": result.get("photo_ratio", ""),
                "expected_ratio": result.get("expected_ratio", ""),
                "min_ratio": result.get("min_ratio", ""),
                "max_ratio": result.get("max_ratio", ""),
            })

            print(
                f"{image_path.name} | "
                f"expected={expected_status} | "
                f"predicted={predicted_status} | "
                f"correct={is_correct} | "
                f"score={result.get('ai_similarity_score'):.6f} | "
                f"threshold={result.get('threshold'):.6f}"
            )

        print("-" * 80)

    accuracy = correct / total if total > 0 else 0

    with open(EVALUATION_RESULT_PATH, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "image_path",
            "file_name",
            "true_label",
            "expected_status",
            "predicted_status",
            "is_correct",
            "ai_similarity_score",
            "threshold",
            "message",
            "face_status",
            "face_count",
            "blur_score",
            "blur_status",
            "brightness_score",
            "brightness_status",
            "background_status",
            "background_avg_std",
            "background_mean_distance",
            "background_outlier_ratio",
            "background_brightness",
            "recommendations",
            "watermark_status",
            "watermark_score",
            "watermark_dark_ratio",
            "watermark_edge_ratio",
            "watermark_white_ratio",
            "ratio_status",
            "photo_width",
            "photo_height",
            "photo_ratio",
            "expected_ratio",
            "min_ratio",
            "max_ratio",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nEvaluasi selesai.")
    print(f"Total data testing : {total}")
    print(f"Prediksi benar     : {correct}")
    print(f"Akurasi sederhana  : {accuracy:.4f}")
    print(f"Hasil CSV disimpan : {EVALUATION_RESULT_PATH}")


if __name__ == "__main__":
    evaluate()