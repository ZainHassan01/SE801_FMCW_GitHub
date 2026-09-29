from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import fft, fftshift
from numpy.lib.stride_tricks import sliding_window_view


ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT / "data_SAAB_SIRS_77GHz_FMCW.npy"
)

OUTPUT_DIR = (
    ROOT / "spectrograms_224x48"
)

CROP_START = 54
CROP_END = 204

WINDOW_LENGTH = 64
HOP_LENGTH = 2
NFFT = 256
DYNAMIC_RANGE_DB = 60.0

# PIL uses (width, height)
OUTPUT_SIZE = (48, 224)

BATCH_SIZE = 16

WINDOW = np.hanning(
    WINDOW_LENGTH
).astype(np.float32)


def generate_spectrograms(signal_batch):
    """
    Input:
        (batch, 5, 256) complex samples

    Output before resizing:
        (batch, 255, 44)

    Output after resizing:
        224 Doppler rows × 48 time columns
    """

    # Retain central 150 slow-time samples
    signal_batch = signal_batch[
        :,
        :,
        CROP_START:CROP_END,
    ]

    # Create 64-sample windows
    frames = sliding_window_view(
        signal_batch,
        window_shape=WINDOW_LENGTH,
        axis=-1,
    )

    # Hop length = 2, producing 44 frames
    frames = frames[
        :,
        :,
        ::HOP_LENGTH,
        :,
    ]

    # Apply Hann window
    frames = frames * WINDOW

    # Complex STFT
    spectra = fft(
        frames,
        n=NFFT,
        axis=-1,
        workers=-1,
    )

    # Centre zero Doppler
    spectra = fftshift(
        spectra,
        axes=-1,
    )

    # Sum power across five range cells
    power = np.sum(
        np.abs(spectra) ** 2,
        axis=1,
    )

    # Convert:
    # (batch, time, frequency)
    # to
    # (batch, frequency, time)
    power = np.transpose(
        power,
        (0, 2, 1),
    )

    # Positive Doppler at top
    power = power[:, ::-1, :]

    # Normalize relative to each image peak
    peak = np.max(
        power,
        axis=(1, 2),
        keepdims=True,
    )

    relative_power = power / np.maximum(
        peak,
        np.finfo(np.float32).tiny,
    )

    # Convert to decibels
    power_db = 10.0 * np.log10(
        np.maximum(
            relative_power,
            1e-6,
        )
    )

    # Clip to −60 to 0 dB
    power_db = np.clip(
        power_db,
        -DYNAMIC_RANGE_DB,
        0.0,
    )

    # Normalize to [0,1]
    images = (
        power_db + DYNAMIC_RANGE_DB
    ) / DYNAMIC_RANGE_DB

    # Remove asymmetric Nyquist row only.
    # No 5% border crop is applied.
    images = images[:, :-1, :]

    return images


def save_image(image, output_path):
    pixels = np.round(
        np.clip(
            image,
            0.0,
            1.0,
        ) * 255.0
    ).astype(np.uint8)

    image = Image.fromarray(pixels)

    image = image.resize(
        OUTPUT_SIZE,
        resample=Image.Resampling.BILINEAR,
    )

    image.save(
        output_path,
        format="PNG",
        optimize=False,
        compress_level=1,
    )


def main():
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading dataset...")

    data = np.load(
        DATASET,
        allow_pickle=True,
    )

    total_created = 0
    total_existing = 0

    for measurement_id in range(
        data.shape[0]
    ):
        signal = np.asarray(
            data[measurement_id, 1]
        )

        if (
            signal.ndim != 2
            or signal.shape[0] != 1280
        ):
            raise ValueError(
                f"Measurement {measurement_id}: "
                f"invalid signal shape {signal.shape}"
            )

        number_of_segments = signal.shape[1]

        measurement_dir = (
            OUTPUT_DIR
            / f"measurement_{measurement_id:03d}"
        )

        measurement_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pending_segments = []

        for segment_id in range(
            number_of_segments
        ):
            output_path = (
                measurement_dir
                / f"segment_{segment_id:05d}.png"
            )

            if output_path.exists():
                total_existing += 1
            else:
                pending_segments.append(
                    segment_id
                )

        for batch_start in range(
            0,
            len(pending_segments),
            BATCH_SIZE,
        ):
            batch_ids = pending_segments[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            signal_batch = (
                signal[:, batch_ids]
                .T
                .reshape(
                    len(batch_ids),
                    5,
                    256,
                )
                .astype(np.complex64)
            )

            images = generate_spectrograms(
                signal_batch
            )

            for index, segment_id in enumerate(
                batch_ids
            ):
                output_path = (
                    measurement_dir
                    / f"segment_{segment_id:05d}.png"
                )

                save_image(
                    images[index],
                    output_path,
                )

                total_created += 1

        print(
            f"Measurement "
            f"{measurement_id + 1:03d}/"
            f"{data.shape[0]} | "
            f"segments={number_of_segments} | "
            f"created={len(pending_segments)}"
        )

    print("\nCompleted")
    print("New images:", total_created)
    print(
        "Existing images skipped:",
        total_existing,
    )
    print(
        "Total images:",
        total_created + total_existing,
    )


if __name__ == "__main__":
    main()