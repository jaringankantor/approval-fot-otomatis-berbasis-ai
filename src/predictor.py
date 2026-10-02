import pickle
import cv2
import numpy as np
from pathlib import Path

from src.config import (
    SVM_MODEL_PATH,
    SCALER_PATH,
    THRESHOLD_PATH
)

from src.feature_extractor import ResNet50FeatureExtractor


class PhotoApprovalPredictor:
    """
    Kelas utama untuk memverifikasi dan memprediksi kelayakan pas foto formal.
    Menggabungkan analisis heuristik citra digital (watermark, rasio, wajah, blur, pencahayaan, latar belakang)
    dengan model machine learning (ResNet50 + One-Class SVM).
    """
    def __init__(self):
        self.extractor = ResNet50FeatureExtractor()

        if not Path(SVM_MODEL_PATH).exists():
            raise FileNotFoundError(f"Model One-Class SVM tidak ditemukan: {SVM_MODEL_PATH}")

        if not Path(SCALER_PATH).exists():
            raise FileNotFoundError(f"Scaler tidak ditemukan: {SCALER_PATH}")

        if not Path(THRESHOLD_PATH).exists():
            raise FileNotFoundError(f"Threshold tidak ditemukan: {THRESHOLD_PATH}")

        with open(SVM_MODEL_PATH, "rb") as file:
            self.model = pickle.load(file)

        with open(SCALER_PATH, "rb") as file:
            self.scaler = pickle.load(file)

        with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
            self.threshold = float(file.read().strip())

    def detect_bottom_watermark(self, image_path):
        """
        Mendeteksi keberadaan watermark, stempel, atau baris teks pada area bawah pas foto.
        Misalnya teks nama sekolah, tanggal, atau atribut cetak lain di bagian bawah foto.
        """
        image = cv2.imread(str(image_path))

        if image is None:
            return {
                "watermark_status": "UNKNOWN",
                "watermark_score": 0,
                "watermark_dark_ratio": "",
                "watermark_edge_ratio": "",
                "watermark_white_ratio": "",
                "message": "File foto tidak dapat dibaca untuk pemeriksaan watermark."
            }

        image = cv2.resize(image, (300, 400))
        h, w, _ = image.shape

        # Membatasi analisis pada area 12% terbawah foto agar pakaian seragam tidak salah terbaca sebagai watermark
        bottom_region = image[int(h * 0.88):h, 0:w]

        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

        # Menghitung rasio piksel gelap (karakter teks umumnya berwarna gelap)
        dark_mask = gray < 95
        dark_ratio = float(np.mean(dark_mask))

        # Menghitung kepadatan kontur tepi untuk mengenali pola teks atau huruf
        edges = cv2.Canny(gray, 70, 160)
        edge_ratio = float(np.mean(edges > 0))

        # Menerapkan segmentasi biner untuk menemukan kandidat komponen teks
        _, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        valid_text_components = 0
        word_like_components = 0
        component_centers_y = []
        component_lefts = []
        component_rights = []

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            fill_ratio = area / (bw * bh)

            # Mengabaikan blok area besar seperti jas, dasi, atau bayangan pakaian
            is_too_large_for_text = (
                bh >= bottom_region.shape[0] * 0.55 or
                bw >= w * 0.45 or
                area >= bottom_region.shape[0] * w * 0.12
            )

            if is_too_large_for_text:
                continue

            # Menyaring komponen berukuran kecil hingga sedang yang memiliki ciri karakter huruf
            is_letter_like = (
                8 <= area <= 650 and
                5 <= bh <= 28 and
                2 <= bw <= 35 and
                0.08 <= fill_ratio <= 0.85
            )

            # Menyaring komponen memanjang yang memiliki ciri gabungan huruf atau kata pendek
            is_word_like = (
                35 < bw <= 140 and
                6 <= bh <= 32 and
                25 <= area <= 1200 and
                0.05 <= fill_ratio <= 0.75
            )

            if is_letter_like:
                valid_text_components += 1
                component_centers_y.append(y + (bh / 2))
                component_lefts.append(x)
                component_rights.append(x + bw)

            if is_word_like:
                word_like_components += 1
                component_centers_y.append(y + (bh / 2))
                component_lefts.append(x)
                component_rights.append(x + bw)

        # Menghitung rasio warna putih terang pada dasar foto (ciri khas banner teks berlatar putih)
        white_ratio = float(np.mean(gray > 225))

        text_line_components = 0
        text_horizontal_span = 0.0

        if len(component_centers_y) > 0:
            bin_count = max(1, bottom_region.shape[0] // 8)
            line_bins = np.clip(
                (np.array(component_centers_y) / bottom_region.shape[0] * bin_count).astype(int),
                0,
                bin_count - 1
            )
            line_counts = np.bincount(line_bins, minlength=bin_count)
            densest_line = int(np.argmax(line_counts))
            line_indexes = np.where(line_bins == densest_line)[0]

            text_line_components = int(line_counts[densest_line])
            line_lefts = [component_lefts[index] for index in line_indexes]
            line_rights = [component_rights[index] for index in line_indexes]
            text_horizontal_span = (max(line_rights) - min(line_lefts)) / w

        is_text_like = (
            valid_text_components >= 6 and
            text_line_components >= 4 and
            text_horizontal_span >= 0.30 and
            edge_ratio >= 0.012
        )

        is_bottom_text_banner = (
            white_ratio >= 0.18 and
            valid_text_components >= 4 and
            text_line_components >= 4 and
            text_horizontal_span >= 0.28 and
            dark_ratio >= 0.006
        )

        is_word_text = (
            word_like_components >= 2 and
            text_line_components >= 2 and
            text_horizontal_span >= 0.30 and
            edge_ratio >= 0.012
        )

        if is_text_like or is_bottom_text_banner or is_word_text:
            return {
                "watermark_status": "DETECTED",
                "watermark_score": int(valid_text_components),
                "watermark_dark_ratio": dark_ratio,
                "watermark_edge_ratio": edge_ratio,
                "watermark_white_ratio": white_ratio,
                "message": "Terdeteksi tulisan/watermark pada bagian bawah foto. Gunakan pas foto tanpa tulisan, logo, atau watermark."
            }

        return {
            "watermark_status": "OK",
            "watermark_score": int(valid_text_components),
            "watermark_dark_ratio": dark_ratio,
            "watermark_edge_ratio": edge_ratio,
            "watermark_white_ratio": white_ratio,
            "message": "Tidak terdeteksi watermark pada bagian bawah foto."
        }
    
    def check_photo_ratio(self, image_path):
        """
        Memeriksa kesesuaian rasio dimensi foto terhadap standar potret 4:3 (lebar : tinggi = 3 : 4 = 0.75).
        Toleransi perbedaan rasio yang diizinkan adalah ±10% (0.675 hingga 0.825).
        """
        image = cv2.imread(str(image_path))

        if image is None:
            return {
                "ratio_status": "INVALID",
                "width": "",
                "height": "",
                "ratio": "",
                "expected_ratio": 0.75,
                "min_ratio": 0.675,
                "max_ratio": 0.825,
                "message": "File foto tidak dapat dibaca untuk pemeriksaan rasio."
            }

        height, width = image.shape[:2]
        ratio = width / height

        expected_ratio = 3 / 4
        tolerance = 0.10

        min_ratio = expected_ratio * (1 - tolerance)
        max_ratio = expected_ratio * (1 + tolerance)

        if min_ratio <= ratio <= max_ratio:
            return {
                "ratio_status": "OK",
                "width": int(width),
                "height": int(height),
                "ratio": float(ratio),
                "expected_ratio": float(expected_ratio),
                "min_ratio": float(min_ratio),
                "max_ratio": float(max_ratio),
                "message": "Rasio foto sesuai format portrait 4x3."
            }

        return {
            "ratio_status": "INVALID",
            "width": int(width),
            "height": int(height),
            "ratio": float(ratio),
            "expected_ratio": float(expected_ratio),
            "min_ratio": float(min_ratio),
            "max_ratio": float(max_ratio),
            "message": "Rasio foto tidak sesuai. Gunakan foto portrait dengan rasio 4x3."
        }

    def detect_face(self, image_path):
        """
        Mendeteksi keberadaan dan jumlah wajah menggunakan Haar Cascade classifier.
        Dikonfigurasi agar toleran terhadap variasi minor namun tetap memastikan hanya ada satu wajah utama.
        """
        image = cv2.imread(str(image_path))

        if image is None:
            return {
                "face_status": "NOT_DETECTED",
                "face_count": 0,
                "message": "File foto tidak dapat dibaca."
            }

        image = cv2.resize(image, (300, 400))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=7,
            minSize=(60, 60)
        )

        h, w = gray.shape
        valid_faces = []

        for (x, y, fw, fh) in faces:
            face_area = fw * fh
            image_area = w * h
            area_ratio = face_area / image_area

            if area_ratio >= 0.03:
                valid_faces.append((x, y, fw, fh, area_ratio))

        if len(valid_faces) == 0:
            return {
                "face_status": "NOT_DETECTED",
                "face_count": 0,
                "message": "Wajah tidak terdeteksi dengan jelas. Pastikan wajah menghadap kamera dan tidak tertutup."
            }

        valid_faces = sorted(valid_faces, key=lambda item: item[4], reverse=True)

        main_area_ratio = valid_faces[0][4]
        significant_faces = []

        for face in valid_faces:
            area_ratio = face[4]

            if area_ratio >= main_area_ratio * 0.6:
                significant_faces.append(face)

        if len(significant_faces) > 1:
            return {
                "face_status": "MULTIPLE_FACES",
                "face_count": len(significant_faces),
                "message": "Terdeteksi lebih dari satu wajah besar. Gunakan foto dengan satu orang saja."
            }

        return {
            "face_status": "OK",
            "face_count": 1,
            "message": "Wajah utama terdeteksi."
        }

    def calculate_blur_score(self, image_path):
        """
        Menghitung nilai skor ketajaman foto menggunakan variansi operator Laplacian.
        Semakin rendah nilainya, semakin buram (blur) citra foto tersebut.
        """
        image = cv2.imread(str(image_path))

        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        return float(blur_score)

    def analyze_blur(self, blur_score):
        """
        Mengevaluasi tingkat ketajaman foto berdasarkan nilai ambang batas blur.
        """
        if blur_score < 50:
            return {
                "blur_status": "BLUR",
                "message": "Foto terdeteksi buram. Gunakan foto yang lebih tajam dan tidak pecah."
            }

        if blur_score < 80:
            return {
                "blur_status": "SLIGHTLY_BLUR",
                "message": "Foto agak kurang tajam. Disarankan menggunakan foto dengan kualitas lebih jelas."
            }

        return {
            "blur_status": "OK",
            "message": "Ketajaman foto cukup baik."
        }

    def calculate_brightness_score(self, image_path):
        """
        Menghitung intensitas rata-rata pencahayaan foto dalam format grayscale (0-255).
        """
        image = cv2.imread(str(image_path))

        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness_score = np.mean(gray)

        return float(brightness_score)

    def analyze_brightness(self, brightness_score):
        """
        Mengevaluasi kesesuaian tingkat pencahayaan foto (terlalu gelap, terlalu terang, atau memadai).
        """
        if brightness_score < 70:
            return {
                "brightness_status": "TOO_DARK",
                "message": "Pencahayaan foto terlalu gelap. Gunakan pencahayaan yang lebih terang dan merata."
            }

        if brightness_score > 210:
            return {
                "brightness_status": "TOO_BRIGHT",
                "message": "Pencahayaan foto terlalu terang. Hindari cahaya berlebihan pada wajah atau background."
            }

        return {
            "brightness_status": "OK",
            "message": "Pencahayaan foto cukup baik."
        }

    def check_background_simple(self, image_path):
        """
        Menganalisis keseragaman warna latar belakang (background) pada area atas foto:
        - Pojok kiri atas
        - Pojok kanan atas
        - Area tepi atas
        """
        image = cv2.imread(str(image_path))

        if image is None:
            return {
                "background_status": "NOT_UNIFORM",
                "background_avg_std": "",
                "background_mean_distance": "",
                "background_outlier_ratio": "",
                "background_brightness": "",
                "message": "File foto tidak dapat dibaca untuk pemeriksaan background."
            }

        image = cv2.resize(image, (300, 400))
        h, w, _ = image.shape

        top_left = image[0:80, 0:80, :]
        top_right = image[0:80, w-80:w, :]
        top_edge = image[0:50, 80:w-80, :]

        patches = [
            top_left,
            top_right,
            top_edge
        ]

        pixels = np.concatenate(
            [patch.reshape(-1, 3) for patch in patches],
            axis=0
        )

        mean_color = np.mean(pixels, axis=0)
        color_std = np.std(pixels, axis=0)
        avg_std = float(np.mean(color_std))

        distances = np.linalg.norm(pixels - mean_color, axis=1)
        mean_distance = float(np.mean(distances))
        outlier_ratio = float(np.mean(distances > 45))

        gray_pixels = cv2.cvtColor(
            pixels.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2GRAY
        )
        background_brightness = float(np.mean(gray_pixels))

        is_uniform = (
            avg_std <= 35 and
            mean_distance <= 40 and
            outlier_ratio <= 0.20
        )

        if is_uniform:
            return {
                "background_status": "OK",
                "background_avg_std": avg_std,
                "background_mean_distance": mean_distance,
                "background_outlier_ratio": outlier_ratio,
                "background_brightness": background_brightness,
                "message": "Background bagian atas terdeteksi cukup seragam untuk pas foto."
            }

        return {
            "background_status": "NOT_UNIFORM",
            "background_avg_std": avg_std,
            "background_mean_distance": mean_distance,
            "background_outlier_ratio": outlier_ratio,
            "background_brightness": background_brightness,
            "message": "Background bagian atas foto terdeteksi kurang seragam. Gunakan background polos sesuai ketentuan."
        }

    def analyze_rejection_reasons(
        self,
        image_path,
        watermark_result=None,
        ratio_result=None,
        ai_rejected=False
    ):
        """
        Menjalankan analisis diagnostik lanjutan ketika foto berstatus REJECTED.
        Hanya indikator yang bermasalah yang akan dimasukkan ke dalam daftar rekomendasi perbaikan.
        """
        if watermark_result is None:
            watermark_result = self.detect_bottom_watermark(image_path)

        if ratio_result is None:
            ratio_result = self.check_photo_ratio(image_path)

        face_result = self.detect_face(image_path)

        blur_score = self.calculate_blur_score(image_path)
        blur_result = self.analyze_blur(blur_score)

        brightness_score = self.calculate_brightness_score(image_path)
        brightness_result = self.analyze_brightness(brightness_score)

        background_result = self.check_background_simple(image_path)

        recommendations = []

        # 1. Rekomendasi watermark: hanya ditambahkan jika terdeteksi watermark/teks
        if watermark_result["watermark_status"] == "DETECTED":
            recommendations.append(watermark_result["message"])

        # 2. Rekomendasi rasio: hanya ditambahkan jika rasio foto tidak sesuai format 4:3
        if ratio_result["ratio_status"] != "OK":
            recommendations.append(ratio_result["message"])

        # 3. Rekomendasi pose/postur: ditambahkan jika model AI menolak foto
        if ai_rejected:
            recommendations.append("Sesuaikan pose sesuai dengan contoh.")

        # 4. Rekomendasi deteksi wajah: hanya ditambahkan jika deteksi wajah tidak normal
        if face_result["face_status"] != "OK":
            recommendations.append(face_result["message"])

        # 5. Rekomendasi ketajaman: hanya ditambahkan jika foto dinilai buram
        if blur_result["blur_status"] != "OK":
            recommendations.append(blur_result["message"])

        # 6. Rekomendasi pencahayaan: hanya ditambahkan jika foto terlalu gelap atau terlalu terang
        if brightness_result["brightness_status"] != "OK":
            recommendations.append(brightness_result["message"])

        # 7. Rekomendasi latar belakang: hanya ditambahkan jika background tidak seragam
        if background_result["background_status"] != "OK":
            recommendations.append(background_result["message"])

        # Jika seluruh pemeriksaan heuristik lolos namun model AI menolak, berikan pesan penolakan umum
        if len(recommendations) == 0:
            recommendations.append(
                "Foto ditolak oleh AI karena tidak sesuai dengan pola foto approved yang dipelajari sistem."
            )

        return {
            "face_status": face_result["face_status"],
            "face_count": face_result["face_count"],

            "blur_score": float(blur_score),
            "blur_status": blur_result["blur_status"],

            "brightness_score": float(brightness_score),
            "brightness_status": brightness_result["brightness_status"],

            "background_status": background_result["background_status"],
            "background_avg_std": background_result.get("background_avg_std", ""),
            "background_mean_distance": background_result.get("background_mean_distance", ""),
            "background_outlier_ratio": background_result.get("background_outlier_ratio", ""),
            "background_brightness": background_result.get("background_brightness", ""),

            "watermark_status": watermark_result["watermark_status"],
            "watermark_score": watermark_result.get("watermark_score", ""),
            "watermark_dark_ratio": watermark_result.get("watermark_dark_ratio", ""),
            "watermark_edge_ratio": watermark_result.get("watermark_edge_ratio", ""),
            "watermark_white_ratio": watermark_result.get("watermark_white_ratio", ""),

            "ratio_status": ratio_result["ratio_status"],
            "photo_width": ratio_result.get("width", ""),
            "photo_height": ratio_result.get("height", ""),
            "photo_ratio": ratio_result.get("ratio", ""),
            "expected_ratio": ratio_result.get("expected_ratio", ""),
            "min_ratio": ratio_result.get("min_ratio", ""),
            "max_ratio": ratio_result.get("max_ratio", ""),

            "recommendations": recommendations
        }

    def predict(self, image_path):
        """
        Menjalankan alur lengkap verifikasi foto:
        1. Validasi keberadaan watermark atau teks tambahan pada bagian bawah.
        2. Validasi rasio aspek dimensi potret 4:3.
        3. Inferensi kesesuaian pola fitur visual menggunakan model One-Class SVM.
        4. Menggabungkan hasil validasi menjadi status akhir (APPROVED / REJECTED).
        5. Melakukan analisis diagnostik mendalam jika foto ditolak.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"File foto tidak ditemukan: {image_path}")

        # Tahap 1: Pemeriksaan keberadaan teks/watermark pada area bawah
        watermark_result = self.detect_bottom_watermark(image_path)
        watermark_detected = watermark_result["watermark_status"] == "DETECTED"

        # Tahap 2: Pemeriksaan kesesuaian rasio dimensi potret 4:3
        ratio_result = self.check_photo_ratio(image_path)
        ratio_invalid = ratio_result["ratio_status"] != "OK"

        # Tahap 3: Inferensi pola kesesuaian pas foto dengan ResNet50 + One-Class SVM
        feature = self.extractor.extract(str(image_path))
        feature = np.array(feature).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature)

        ai_score = self.model.decision_function(feature_scaled)[0]
        ai_status = "APPROVED" if ai_score >= self.threshold else "REJECTED"

        # Tahap 4: Penentuan keputusan status akhir
        # Adanya watermark atau rasio yang tidak sesuai akan langsung menyebabkan status REJECTED
        rejection_sources = []

        if watermark_detected:
            rejection_sources.append("WATERMARK")

        if ratio_invalid:
            rejection_sources.append("RATIO")

        if ai_status == "REJECTED":
            rejection_sources.append("AI")

        if len(rejection_sources) > 0:
            final_status = "REJECTED"
        else:
            final_status = "APPROVED"

        result = {
            "image_path": str(image_path),
            "status": final_status,
            "ai_status": ai_status,
            "rejection_source": ",".join(rejection_sources),
            "ai_similarity_score": float(ai_score),
            "threshold": float(self.threshold)
        }

        # Tahap 5: Jika status akhir APPROVED, kembalikan hasil ringkas tanpa rincian penolakan
        if final_status == "APPROVED":
            result["message"] = "Foto diterima berdasarkan pemeriksaan format dan pola foto approved yang dipelajari AI."
            result["recommendations"] = [
                "Foto memenuhi pola approved."
            ]
            return result

        # Tahap 6: Jika status akhir REJECTED, sertakan analisis diagnostik dan rekomendasi perbaikan
        rejection_analysis = self.analyze_rejection_reasons(
            image_path=image_path,
            watermark_result=watermark_result,
            ratio_result=ratio_result,
            ai_rejected=(ai_status == "REJECTED")
        )

        result.update(rejection_analysis)

        if watermark_detected:
            result["message"] = "Foto ditolak karena terdeteksi tulisan/watermark pada bagian bawah foto."
        elif ratio_invalid:
            result["message"] = "Foto ditolak karena rasio foto tidak sesuai format portrait 4x3."
        elif ai_status == "REJECTED":
            result["message"] = "Foto ditolak berdasarkan pola AI. Berikut analisis kemungkinan penyebab penolakan."
        else:
            result["message"] = "Foto ditolak karena tidak memenuhi ketentuan format."

        return result
