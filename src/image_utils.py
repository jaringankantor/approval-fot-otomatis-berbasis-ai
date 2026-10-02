from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob('*') if p.suffix.lower() in VALID_EXTENSIONS])


def load_rgb_image(path: Path, size: Tuple[int, int] = (300, 400)) -> Image.Image:
    image = Image.open(path).convert('RGB')
    # ImageOps.exif_transpose mencegah foto dari HP terbaca miring karena metadata EXIF
    image = ImageOps.exif_transpose(image)
    if image.size != size:
        image = image.resize(size)
    return image


def rgb_to_gray_np(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image).astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return gray


def laplacian_variance(image: Image.Image) -> float:
    """Skor blur sederhana: makin kecil nilainya, makin buram."""
    gray = rgb_to_gray_np(image)
    # Kernel Laplacian 3x3 tanpa OpenCV
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
    return float(np.mean(rgb_to_gray_np(image)))


def border_pixels(image: Image.Image, border_ratio: float = 0.08) -> np.ndarray:
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
    pixels = border_pixels(image)
    return {
        'mean_rgb': pixels.mean(axis=0).tolist(),
        'std_rgb': pixels.std(axis=0).tolist(),
        'border_std_mean': float(pixels.std(axis=0).mean())
    }
