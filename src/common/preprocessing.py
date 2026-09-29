"""SE-801 shared preprocessing for native 255x44 radar STFT PNGs.

The PNG is stored at native STFT geometry. Grayscale, geometric cropping,
bilinear resizing, and fold-specific standardization happen here at load time.
Image dimensions are (height, width) = (Doppler, time).
"""

from pathlib import Path

import numpy as np
from PIL import Image


NATIVE_WIDTH = 44
NATIVE_HEIGHT = 255
OUTPUT_WIDTH = 224
OUTPUT_HEIGHT = 224
BORDER_FRACTION = 0.05


def preprocess_image(path: str | Path, *, mean: float | None = None,
                     std: float | None = None) -> np.ndarray:
    """Return float32 grayscale as (1, 224, 224).

    With mean/std omitted, values are in [0, 1] for train-fold statistics and
    image-derived features. Supply BOTH training-fold mean and std when loading
    model inputs from training, validation, or test partitions.
    """
    if (mean is None) != (std is None):
        raise ValueError("mean and std must both be supplied or both omitted")

    with Image.open(path) as source:
        if source.size != (NATIVE_WIDTH, NATIVE_HEIGHT):
            raise ValueError(
                f"Expected native STFT PNG {NATIVE_WIDTH}x{NATIVE_HEIGHT}; "
                f"found {source.size} in {path}. Do not crop an already resized image."
            )
        image = source.convert("L")

    left = right = round(NATIVE_WIDTH * BORDER_FRACTION)    # 2 columns
    top = bottom = round(NATIVE_HEIGHT * BORDER_FRACTION)  # 13 rows
    image = image.crop((left, top, NATIVE_WIDTH - right, NATIVE_HEIGHT - bottom))
    assert image.size == (40, 229)
    image = image.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.BILINEAR)

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    if mean is not None:
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("Invalid training-fold mean or standard deviation")
        pixels = (pixels - mean) / std
    return pixels[np.newaxis, ...].astype(np.float32, copy=False)
