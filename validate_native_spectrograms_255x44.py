"""Validate the 75,868 native 44x255 grayscale STFT PNGs."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "spectrograms_native_255x44"
FOLD_MANIFEST = ROOT / "manifests_224x48" / "fold_manifest.csv"
OUTPUT_ROOT = ROOT / "manifests_224x224"
EXPECTED_COUNT = 75_868
EXPECTED_SIZE = (44, 255)  # PIL: width x height
PATTERN = re.compile(r"(?:^|/)(measurement_\d+)/(segment_\d+\.png)$")


def expected_paths() -> list[Path]:
    folds = pd.read_csv(FOLD_MANIFEST, usecols=["filename", "outer fold"])
    names = (
        folds.loc[folds["outer fold"] == 1, "filename"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    if len(names) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNT} unique manifest images; found {len(names)}"
        )

    paths = []
    for name in names:
        match = PATTERN.search(name.replace("\\", "/"))
        if match is None:
            raise RuntimeError(f"Unsupported manifest filename: {name}")
        paths.append(IMAGE_ROOT / match.group(1) / match.group(2))

    if len(set(paths)) != EXPECTED_COUNT:
        raise RuntimeError("Manifest maps multiple rows to the same native PNG")
    return sorted(paths)


def create_montage(paths: list[Path], destination: Path) -> None:
    indexes = np.linspace(0, len(paths) - 1, 16, dtype=int)
    cell_width, cell_height = 180, 300
    canvas = Image.new("L", (4 * cell_width, 4 * cell_height), color=0)
    draw = ImageDraw.Draw(canvas)
    for slot, index in enumerate(indexes):
        path = paths[int(index)]
        with Image.open(path) as source:
            preview = source.convert("L").resize((52, 255), Image.Resampling.NEAREST)
        x = (slot % 4) * cell_width + 64
        y = (slot // 4) * cell_height + 5
        canvas.paste(preview, (x, y))
        label = f"{path.parent.name}\n{path.name}"
        draw.multiline_text(((slot % 4) * cell_width + 5, y + 260), label, fill=255)
    canvas.save(destination, format="PNG")


def main() -> None:
    if not FOLD_MANIFEST.is_file():
        raise FileNotFoundError(FOLD_MANIFEST)
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(IMAGE_ROOT)

    expected = expected_paths()
    expected_set = set(expected)
    actual = set(IMAGE_ROOT.rglob("*.png"))
    missing = sorted(expected_set - actual)
    extra = sorted(actual - expected_set)

    invalid = []
    constant = []
    global_min, global_max = 255, 0
    for number, path in enumerate(expected, start=1):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode != "L" or image.size != EXPECTED_SIZE or image.format != "PNG":
                    invalid.append({
                        "path": str(path.relative_to(ROOT)),
                        "mode": image.mode,
                        "size": list(image.size),
                        "format": image.format,
                    })
                    continue
                pixels = np.asarray(image)
            low, high = int(pixels.min()), int(pixels.max())
            global_min = min(global_min, low)
            global_max = max(global_max, high)
            if low == high:
                constant.append(str(path.relative_to(ROOT)))
        except Exception as error:
            invalid.append({
                "path": str(path.relative_to(ROOT)),
                "error": f"{type(error).__name__}: {error}",
            })
        if number % 5000 == 0:
            print(f"Validated {number}/{EXPECTED_COUNT}", flush=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_ROOT / "native_spectrogram_validation.json"
    montage_path = OUTPUT_ROOT / "native_spectrogram_QA_montage.png"
    passed = not missing and not extra and not invalid and not constant
    report = {
        "status": "PASS" if passed else "FAIL",
        "expected_images": EXPECTED_COUNT,
        "actual_png_files": len(actual),
        "geometry": {"width": 44, "height": 255},
        "mode": "L",
        "pixel_range_across_dataset": [global_min, global_max],
        "missing_count": len(missing),
        "extra_count": len(extra),
        "invalid_count": len(invalid),
        "constant_count": len(constant),
        "missing_first_20": [str(p.relative_to(ROOT)) for p in missing[:20]],
        "extra_first_20": [str(p.relative_to(ROOT)) for p in extra[:20]],
        "invalid_first_20": invalid[:20],
        "constant_first_20": constant[:20],
        "processing_note": (
            "These are native STFT PNGs. The common pipeline must crop 5% "
            "from every side before bilinear resizing to 224x224."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not missing and expected:
        create_montage(expected, montage_path)

    print(f"\nValidation: {report['status']}")
    print(f"Images: {len(actual)}")
    print(f"Missing: {len(missing)}")
    print(f"Extra: {len(extra)}")
    print(f"Invalid: {len(invalid)}")
    print(f"Constant images: {len(constant)}")
    print(f"Report: {report_path}")
    print(f"Montage: {montage_path}")
    if not passed:
        raise RuntimeError("NATIVE_SPECTROGRAM_VALIDATION_FAIL")
    print("NATIVE_SPECTROGRAM_VALIDATION_PASS")


if __name__ == "__main__":
    main()
