import argparse
import csv
import numpy as np

from src.config import (
    EVALUATION_RESULT_PATH,
    RESULTS_DIR,
    THRESHOLD_PATH
)


def read_scores_from_evaluation():
    """
    Membaca skor kemiripan AI (ai_similarity_score) dari file CSV hasil evaluasi,
    lalu mengelompokkannya ke dalam array approved dan rejected.
    """
    if not EVALUATION_RESULT_PATH.exists():
        raise FileNotFoundError(
            f"File evaluasi tidak ditemukan: {EVALUATION_RESULT_PATH}\n"
            "Jalankan dulu: python 04_evaluate_testing.py"
        )

    approved_scores = []
    rejected_scores = []

    with open(EVALUATION_RESULT_PATH, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            score_text = row.get("ai_similarity_score")

            if score_text is None or score_text == "":
                continue

            score = float(score_text)
            expected_status = row.get("expected_status")

            if expected_status == "APPROVED":
                approved_scores.append(score)
            elif expected_status == "REJECTED":
                rejected_scores.append(score)

    if len(approved_scores) == 0:
        raise ValueError("Tidak ada score untuk data approved pada hasil evaluasi.")

    if len(rejected_scores) == 0:
        print("Peringatan: tidak ada score untuk data rejected pada hasil evaluasi.")

    return np.array(approved_scores), np.array(rejected_scores)


def evaluate_threshold(threshold, approved_scores, rejected_scores):
    """
    Menghitung metrik performa (akurasi, false accept, false reject) untuk suatu kandidat threshold.
    """
    approved_correct = np.sum(approved_scores >= threshold)
    rejected_correct = np.sum(rejected_scores < threshold)

    total = len(approved_scores) + len(rejected_scores)
    correct = approved_correct + rejected_correct

    accuracy = correct / total if total > 0 else 0

    false_reject = len(approved_scores) - approved_correct
    false_accept = len(rejected_scores) - rejected_correct

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "approved_correct": int(approved_correct),
        "rejected_correct": int(rejected_correct),
        "false_reject": int(false_reject),
        "false_accept": int(false_accept)
    }


def main():
    """
    Mencari nilai ambang batas optimal yang memaksimalkan akurasi klasifikasi pada dataset evaluasi,
    serta menyediakan opsi untuk memperbarui file model threshold.
    """
    parser = argparse.ArgumentParser(
        description="Mencari kandidat threshold berdasarkan hasil evaluasi testing."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Tulis threshold terbaik ke models/threshold.txt. Tanpa opsi ini, threshold hanya dilaporkan."
    )
    args = parser.parse_args()

    approved_scores, rejected_scores = read_scores_from_evaluation()

    all_scores = np.concatenate([approved_scores, rejected_scores])

    min_score = float(np.min(all_scores))
    max_score = float(np.max(all_scores))

    candidate_thresholds = np.linspace(min_score, max_score, 100)

    best_result = None

    print("Mencari threshold terbaik berdasarkan hasil evaluasi...\n")
    print(f"Jumlah approved score : {len(approved_scores)}")
    print(f"Jumlah rejected score : {len(rejected_scores)}")
    print(f"Score minimum         : {min_score}")
    print(f"Score maksimum        : {max_score}")

    for threshold in candidate_thresholds:
        result = evaluate_threshold(threshold, approved_scores, rejected_scores)

        if best_result is None:
            best_result = result
        elif result["accuracy"] > best_result["accuracy"]:
            best_result = result
        elif result["accuracy"] == best_result["accuracy"]:
            # Jika nilai akurasi sama, utamakan threshold dengan false accept terendah
            if result["false_accept"] < best_result["false_accept"]:
                best_result = result

    print("\nThreshold terbaik ditemukan:")
    print(f"Threshold     : {best_result['threshold']}")
    print(f"Akurasi       : {best_result['accuracy']:.4f}")
    print(f"False reject  : {best_result['false_reject']}")
    print(f"False accept  : {best_result['false_accept']}")
    print(f"Approved benar: {best_result['approved_correct']}")
    print(f"Rejected benar: {best_result['rejected_correct']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tuned_threshold_path = RESULTS_DIR / "tuned_threshold.txt"

    with open(tuned_threshold_path, "w", encoding="utf-8") as file:
        file.write(str(best_result["threshold"]))

    print(f"\nKandidat threshold disimpan ke: {tuned_threshold_path}")

    if not args.apply:
        print("\nmodels/threshold.txt tidak diubah.")
        print("Threshold produksi tetap berasal dari validation pada 03_train_oneclass_svm.py.")
        print("Jika benar-benar ingin menerapkan hasil tuning ini, jalankan:")
        print("python 05_tune_threshold.py --apply")
        return

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as file:
        file.write(str(best_result["threshold"]))

    print(f"\nThreshold baru disimpan ke: {THRESHOLD_PATH}")
    print("\nJalankan ulang evaluasi untuk melihat hasil setelah tuning:")
    print("python 04_evaluate_testing.py")


if __name__ == "__main__":
    main()
