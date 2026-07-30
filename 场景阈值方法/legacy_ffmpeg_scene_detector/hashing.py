"""Perceptual hashing helpers for the archived scene detector."""

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
    """Return a 64-bit perceptual hash as a boolean array."""
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


def frame_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Mean absolute pixel difference normalized to the range 0..1."""
    if first.shape != second.shape:
        raise ValueError("frames must have identical shapes")
    first_float = first.astype(np.float32, copy=False)
    second_float = second.astype(np.float32, copy=False)
    return float(np.mean(np.abs(first_float - second_float)) / 255.0)


def compare_grid(
    before: np.ndarray,
    after: np.ndarray,
    *,
    rows: int,
    columns: int,
    changed_distance: int,
) -> tuple[float, float, list[int]]:
    """Compare corresponding grid blocks using pHash.

    Returns (changed block ratio, mean Hamming distance, all distances).
    """
    if before.shape != after.shape:
        raise ValueError("frames must have identical shapes")
    if before.ndim not in (2, 3):
        raise ValueError("frames must be grayscale or RGB arrays")
    height, width = before.shape[:2]
    if height < rows or width < columns:
        raise ValueError("grid cannot be larger than the frame")

    before_hashes = grid_hashes(before, rows=rows, columns=columns)
    after_hashes = grid_hashes(after, rows=rows, columns=columns)
    distances = [
        hamming_distance(first, second)
        for first, second in zip(before_hashes, after_hashes)
    ]
    changed = sum(distance >= changed_distance for distance in distances)
    ratio = changed / len(distances)
    return ratio, float(np.mean(distances)), distances


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
        y0 = row * image.shape[0] // rows
        y1 = (row + 1) * image.shape[0] // rows
        for column in range(columns):
            x0 = column * image.shape[1] // columns
            x1 = (column + 1) * image.shape[1] // columns
            hashes.append(phash(image[y0:y1, x0:x1]))
    return np.stack(hashes)
