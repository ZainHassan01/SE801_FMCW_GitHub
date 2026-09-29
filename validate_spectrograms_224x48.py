import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT / "data_SAAB_SIRS_77GHz_FMCW.npy"
)

SPECTROGRAM_DIR = (
    ROOT / "spectrograms_224x48"
)

OUTPUT_DIR = (
    ROOT / "manifests_224x48"
)

EXPECTED_SIZE = (48, 224)
EXPECTED_IMAGES = 75868
SAMPLE_SIZE = 1000


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading source dataset...")

    data = np.load(
        DATASET,
        allow_pickle=True,
    )

    expected_files = set()

    for measurement_id in range(data.shape[0]):
        signal = np.asarray(
            data[measurement_id, 1]
        )

        for segment_id in range(signal.shape[1]):
            expected_files.add(
                (
                    SPECTROGRAM_DIR
                    / f"measurement_{measurement_id:03d}"
                    / f"segment_{segment_id:05d}.png"
                ).resolve()
            )

    del data

    actual_paths = sorted(
        path.resolve()
        for path in SPECTROGRAM_DIR.rglob("*.png")
    )

    actual_files = set(actual_paths)

    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files

    invalid_files = []

    for index, path in enumerate(
        actual_paths,
        start=1,
    ):
        try:
            with Image.open(path) as image:
                if image.size != EXPECTED_SIZE:
                    invalid_files.append(
                        f"{path}: size={image.size}"
                    )

                if image.mode != "L":
                    invalid_files.append(
                        f"{path}: mode={image.mode}"
                    )

                image.verify()

        except Exception as error:
            invalid_files.append(
                f"{path}: {error}"
            )

        if index % 5000 == 0:
            print(
                f"Validated {index}/"
                f"{len(actual_paths)}"
            )

    sample_count = min(
        SAMPLE_SIZE,
        len(actual_paths),
    )

    sample_indices = np.linspace(
        0,
        len(actual_paths) - 1,
        sample_count,
        dtype=int,
    )

    means = []
    standard_deviations = []
    minimums = []
    maximums = []
    constant_images = []

    for index in sample_indices:
        path = actual_paths[index]

        with Image.open(path) as image:
            pixels = np.asarray(
                image,
                dtype=np.float32,
            ) / 255.0

        means.append(float(pixels.mean()))
        standard_deviations.append(
            float(pixels.std())
        )
        minimums.append(float(pixels.min()))
        maximums.append(float(pixels.max()))

        if pixels.std() == 0:
            constant_images.append(str(path))

    montage_indices = np.linspace(
        0,
        len(actual_paths) - 1,
        16,
        dtype=int,
    )

    montage = Image.new(
        "L",
        (
            EXPECTED_SIZE[0] * 8,
            EXPECTED_SIZE[1] * 2,
        ),
    )

    for position, index in enumerate(
        montage_indices
    ):
        with Image.open(
            actual_paths[index]
        ) as image:
            column = position % 8
            row = position // 8

            montage.paste(
                image,
                (
                    column * EXPECTED_SIZE[0],
                    row * EXPECTED_SIZE[1],
                ),
            )

    montage_path = (
        OUTPUT_DIR
        / "spectrogram_QA_montage.png"
    )

    montage.save(montage_path)

    passed = (
        len(actual_paths) == EXPECTED_IMAGES
        and not missing_files
        and not extra_files
        and not invalid_files
        and not constant_images
    )

    report = {
        "passed": passed,
        "expected_images": EXPECTED_IMAGES,
        "actual_images": len(actual_paths),
        "expected_size_width_height": list(
            EXPECTED_SIZE
        ),
        "missing_files": len(missing_files),
        "extra_files": len(extra_files),
        "invalid_files": invalid_files[:20],
        "sample_size": sample_count,
        "sample_pixel_mean": float(
            np.mean(means)
        ),
        "sample_pixel_std_mean": float(
            np.mean(standard_deviations)
        ),
        "sample_pixel_min": float(
            np.min(minimums)
        ),
        "sample_pixel_max": float(
            np.max(maximums)
        ),
        "constant_sample_images": (
            constant_images
        ),
    }

    report_path = (
        OUTPUT_DIR
        / "spectrogram_validation.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        "\nValidation:",
        "PASS" if passed else "FAIL",
    )
    print("Images:", len(actual_paths))
    print("Missing:", len(missing_files))
    print("Extra:", len(extra_files))
    print("Invalid:", len(invalid_files))
    print(
        "Constant sample images:",
        len(constant_images),
    )
    print("Report:", report_path)
    print("Montage:", montage_path)


if __name__ == "__main__":
    main()