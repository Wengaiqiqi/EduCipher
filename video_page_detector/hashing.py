from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image


@lru_cache(maxsize=8)
def _dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * size))
    matrix[0, :] *= np.sqrt(1.0 / size)
    matrix[1:, :] *= np.sqrt(2.0 / size)
    return matrix


def _as_gray_array(image: Image.Image | np.ndarray, size: int) -> np.ndarray:
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8, copy=False))
    else:
        pil_image = image
    resized = pil_image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float64)


def phash(
    image: Image.Image | np.ndarray,
    *,
    sample_size: int = 32,
    hash_size: int = 8,
) -> np.ndarray:
    if hash_size > sample_size:
        raise ValueError("hash_size cannot exceed sample_size")
    pixels = _as_gray_array(image, sample_size)
    transform = _dct_matrix(sample_size)
    coefficients = transform @ pixels @ transform.T
    low_frequency = coefficients[:hash_size, :hash_size]
    values_without_dc = low_frequency.reshape(-1)[1:]
    median = float(np.median(values_without_dc))
    return (low_frequency > median).reshape(-1)


def hamming_distance(first: np.ndarray, second: np.ndarray) -> int:
    if first.shape != second.shape:
        raise ValueError("hashes must have identical shapes")
    return int(np.count_nonzero(first != second))


def grid_hashes(
    image: np.ndarray,
    *,
    rows: int,
    columns: int,
) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError("image must be grayscale or RGB")
    height, width = image.shape[:2]
    if height < rows or width < columns:
        raise ValueError("grid cannot be larger than the image")
    hashes: list[np.ndarray] = []
    for row in range(rows):
        y0 = row * height // rows
        y1 = (row + 1) * height // rows
        for column in range(columns):
            x0 = column * width // columns
            x1 = (column + 1) * width // columns
            hashes.append(phash(image[y0:y1, x0:x1]))
    return np.stack(hashes)
