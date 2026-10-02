from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def list_images(folder: Path) -> List[Path]:
    """
    Mengambil dan mengurutkan seluruh path file gambar dari direktori yang ditentukan.
    """
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob('*') if p.suffix.lower() in VALID_EXTENSIONS])


def load_rgb_image(path: Path, size: Tuple[int, int] = (300, 400)) -> Image.Image:
    """
    Membuka file gambar, menyesuaikan orientasi EXIF, mengubah ke format warna RGB,
    dan mengubah dimensi sesuai target ukuran.
    """
    image = Image.open(path).convert('RGB')
    # Menyesuaikan orientasi gambar berdasarkan metadata EXIF agar orientasi tidak terbalik
    image = ImageOps.exif_transpose(image)
    if image.size != size:
        image = image.resize(size)
    return image


def rgb_to_gray_np(image: Image.Image) -> np.ndarray:
    """
    Mengonversi gambar PIL RGB ke representasi array numpy grayscale
    menggunakan bobot standar luminance (0.299 R + 0.587 G + 0.114 B).
    """
    arr = np.asarray(image).astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return gray


def laplacian_variance(image: Image.Image) -> float:
    """
    Menghitung skor ketajaman gambar menggunakan variansi operator Laplacian (kernel 3x3).
    Semakin kecil nilai skor, semakin buram (blur) gambar tersebut.
    """
    gray = rgb_to_gray_np(image)
    # Operasi konvolusi Laplacian 3x3 secara manual menggunakan padding tepi
    padded = np.pad(gray, 1, mode='edge')
    lap = (
        -4 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(lap))


def brightness_score(image: Image.Image) -> float:
    """
    Menghitung rata-rata intensitas piksel grayscale sebagai representasi tingkat pencahayaan gambar.
    """
    return float(np.mean(rgb_to_gray_np(image)))


def border_pixels(image: Image.Image, border_ratio: float = 0.08) -> np.ndarray:
    """
    Mengambil piksel-piksel pada area tepi (border) gambar (atas, bawah, kiri, kanan)
    untuk analisis keseragaman latar belakang.
    """
    arr = np.asarray(image).astype(np.float32)
    h, w, _ = arr.shape
    bw = max(1, int(w * border_ratio))
    bh = max(1, int(h * border_ratio))
    top = arr[:bh, :, :]
    bottom = arr[-bh:, :, :]
    left = arr[:, :bw, :]
    right = arr[:, -bw:, :]
    return np.concatenate([
        top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)
    ], axis=0)


def background_stats(image: Image.Image) -> dict:
    """
    Menghitung statistik warna (rata-rata dan standar deviasi RGB) pada area tepi foto.
    """
    pixels = border_pixels(image)
    return {
        'mean_rgb': pixels.mean(axis=0).tolist(),
        'std_rgb': pixels.std(axis=0).tolist(),
        'border_std_mean': float(pixels.std(axis=0).mean())
    }
