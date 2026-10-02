import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.config import (
    MODELS_DIR,
    TRAINING_FEATURES_PATH,
    VALIDATION_FEATURES_PATH,
    SVM_MODEL_PATH,
    SCALER_PATH,
    THRESHOLD_PATH
)


# Parameter model One-Class SVM:
# nu: estimasi proporsi pencilan (outlier) yang ditoleransi pada data latih
# gamma: penyesuaian koefisien kernel RBF terhadap variansi fitur
OCSVM_NU = 0.05
OCSVM_GAMMA = "scale"


def main():
    """
    Melatih model One-Class SVM menggunakan fitur data latih yang telah distandardisasi,
    menghitung nilai ambang batas (threshold) dari data validasi, lalu menyimpan model, scaler, dan threshold.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Memuat fitur training dan validation...")

    training_features = np.load(TRAINING_FEATURES_PATH)
    validation_features = np.load(VALIDATION_FEATURES_PATH)

    print(f"Ukuran fitur training   : {training_features.shape}")
    print(f"Ukuran fitur validation : {validation_features.shape}")

    print("\nMelakukan standardisasi fitur...")

    scaler = StandardScaler()
    training_features_scaled = scaler.fit_transform(training_features)
    validation_features_scaled = scaler.transform(validation_features)

    print("\nMelatih One-Class SVM...")

    model = OneClassSVM(
        kernel="rbf",
        nu=OCSVM_NU,
        gamma=OCSVM_GAMMA
    )

    model.fit(training_features_scaled)

    print("Training One-Class SVM selesai.")

    print("\nMenghitung threshold dari data validation approved...")

    validation_scores = model.decision_function(validation_features_scaled)

    # Menghitung ambang batas (threshold) dari persentil ke-5 skor data validasi approved.
    # Nilai persentil yang lebih rendah menghasilkan standar penerimaan yang lebih ketat.
    threshold = np.percentile(validation_scores, 5)

    print(f"Skor validation minimum : {validation_scores.min()}")
    print(f"Skor validation maksimum: {validation_scores.max()}")
    print(f"Skor validation rata-rata: {validation_scores.mean()}")
    print(f"Threshold approval       : {threshold}")

    print("\nMenyimpan model, scaler, dan threshold...")

    with open(SVM_MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    with open(SCALER_PATH, "wb") as file:
        pickle.dump(scaler, file)

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as file:
        file.write(str(threshold))

    print("\nModel berhasil disimpan.")
    print(f"Model One-Class SVM : {SVM_MODEL_PATH}")
    print(f"Scaler              : {SCALER_PATH}")
    print(f"Threshold           : {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()