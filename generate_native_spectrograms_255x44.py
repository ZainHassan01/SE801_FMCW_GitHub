"""Generate native STFT PNGs; leave crop and resize to src/common.

Input: data_SAAB_SIRS_77GHz_FMCW.npy in the project root.
Output: spectrograms_native_255x44/measurement_NNN/segment_NNNNN.png.
Each PNG is 44 pixels wide (time) by 255 pixels high (Doppler).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import stft


PRF_HZ = 17_000
CROP_START = 54
CROP_END = 204  # 150 samples, original sweep indices 54..203
WINDOW = 64
HOP = 2
NFFT = 256
OUTPUT_SHAPE = (255, 44)  # NumPy: rows (Doppler), columns (time)
BATCH_SIZE = 64
FILE_PATTERN = re.compile(r"(?:^|/)(measurement_\d+)/(segment_(\d+)\.png)$")


def filenames_from_manifest(root: Path) -> tuple[dict[int, list[tuple[str, str]]], int]:
    """Preserve old manifest identities even if numbering began at one."""
    manifest = root / "manifests_224x48/fold_manifest.csv"
    folds = pd.read_csv(manifest, usecols=[
        "filename", "original-image identifier", "outer fold"
    ])
    one_outer = folds.loc[folds["outer fold"] == 1].drop_duplicates("filename")
    if len(one_outer) != 75868:
        raise RuntimeError("Fold manifest must have exactly 75,868 unique images in outer fold 1")
    parent_ids = {int(v) for v in one_outer["original-image identifier"]}
    if parent_ids == set(range(130)):
        parent_offset = 0
    elif parent_ids == set(range(1, 131)):
        parent_offset = 1
    else:
        raise RuntimeError("Cannot map original-image identifiers to source measurement rows")

    filenames = {}
    for parent_id, group in one_outer.groupby("original-image identifier"):
        found = []
        for old_path in group["filename"]:
            match = FILE_PATTERN.search(str(old_path).replace("\\", "/"))
            if match is None:
                raise RuntimeError(f"Unrecognized image filename: {old_path}")
            found.append((int(match.group(3)), match.group(1), match.group(2)))
        found.sort(key=lambda row: row[0])
        indices = [row[0] for row in found]
        if indices not in (list(range(len(found))), list(range(1, len(found) + 1))):
            raise RuntimeError(f"Unexpected segment IDs for parent {parent_id}")
        filenames[int(parent_id)] = [(folder, filename) for _, folder, filename in found]
    return filenames, parent_offset


def make_native_images(segments: np.ndarray) -> np.ndarray:
    """Convert (batch, 5, 256) complex segments to (batch, 255, 44) uint8."""
    if segments.ndim != 3 or segments.shape[1:] != (5, 256):
        raise ValueError(f"Expected (batch, 5, 256); got {segments.shape}")
    signal = segments[:, :, CROP_START:CROP_END]
    _, _, spectrum = stft(
        signal,
        fs=PRF_HZ,
        window="hann",
        nperseg=WINDOW,
        noverlap=WINDOW - HOP,
        nfft=NFFT,
        boundary=None,
        padded=False,
        return_onesided=False,
        axis=-1,
    )
    # Spectrum has shape (batch, 5 range cells, 256 frequency bins, 44 frames).
    spectrum = np.fft.fftshift(spectrum, axes=-2)
    # Remove the negative Nyquist row; remaining zero Doppler is central row 127.
    spectrum = spectrum[:, :, 1:, :]
    power = (spectrum.real ** 2 + spectrum.imag ** 2).sum(axis=1)
    if power.shape[1:] != OUTPUT_SHAPE or not np.isfinite(power).all():
        raise RuntimeError(f"Invalid STFT power: {power.shape}")

    # Encode log relative power without learning normalization from held-out folds.
    peak = np.maximum(np.max(power, axis=(1, 2), keepdims=True), 1e-30)
    relative = np.maximum(power / peak, 1e-6)
    db = np.clip(10.0 * np.log10(relative), -60.0, 0.0)
    return np.rint((db + 60.0) * (255.0 / 60.0)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--force", action="store_true", help="Replace existing native PNGs")
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "data_SAAB_SIRS_77GHz_FMCW.npy"
    target = root / "spectrograms_native_255x44"
    if not source.is_file():
        raise FileNotFoundError(source)

    print("Loading source dataset...", flush=True)
    data = np.load(source, allow_pickle=True)
    if data.shape != (130, 6):
        raise RuntimeError(f"Unexpected dataset shape {data.shape}")
    filenames, parent_offset = filenames_from_manifest(root)

    made = skipped = 0
    for measurement_id in range(len(data)):
        matrix = np.asarray(data[measurement_id, 1])
        if matrix.ndim != 2 or matrix.shape[0] != 1280:
            raise RuntimeError(f"Measurement {measurement_id}: expected (1280, N), got {matrix.shape}")
        number = matrix.shape[1]
        names = filenames[measurement_id + parent_offset]
        if len(names) != number or len({name for _, name in names}) != number:
            raise RuntimeError(f"Measurement {measurement_id}: manifest and source segment counts disagree")
        folders = {folder for folder, _ in names}
        if len(folders) != 1:
            raise RuntimeError(f"Measurement {measurement_id}: manifest uses multiple folders")
        folder = target / next(iter(folders))
        folder.mkdir(parents=True, exist_ok=True)

        for start in range(0, number, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, number)
            pending = []
            for segment_id in range(start, stop):
                path = folder / names[segment_id][1]
                if path.is_file() and not args.force:
                    with Image.open(path) as old:
                        if old.size != (44, 255) or old.mode != "L":
                            raise RuntimeError(f"Wrong existing image geometry: {path}")
                    skipped += 1
                else:
                    pending.append(segment_id)
            if not pending:
                continue
            # 1280 values = five separate range cells x 256 slow-time sweeps.
            samples = matrix[:, pending].T.reshape(len(pending), 5, 256)
            pngs = make_native_images(samples)
            for segment_id, png in zip(pending, pngs):
                path = folder / names[segment_id][1]
                temporary = path.with_suffix(".tmp")
                Image.fromarray(png).save(temporary, format="PNG")
                temporary.replace(path)
                made += 1
        print(f"Measurement {measurement_id + 1:03d}/130 | segments={number} | created={made} | skipped={skipped}", flush=True)

    if made + skipped != 75868:
        raise RuntimeError(f"Unexpected total {made + skipped}; expected 75868")
    print(f"NATIVE_STFT_PASS | total={made + skipped} | new={made} | reused={skipped}")
    print(f"Native PNG folder: {target}")


if __name__ == "__main__":
    main()
