from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import fft, fftshift
from numpy.lib.stride_tricks import sliding_window_view


DATASET = Path(
    r"C:\Users\Eng Zain\Desktop\Spring 26\ann"
    r"\data_SAAB_SIRS_77GHz_FMCW.npy"
)

OUTPUT_DIR = Path("spectrograms")

CROP_START = 54
CROP_END = 204

WINDOW_LENGTH = 64
HOP_LENGTH = 2
NFFT = 256
DYNAMIC_RANGE_DB = 60.0

OUTPUT_SIZE = (224, 224)
BATCH_SIZE = 16

WINDOW = np.hanning(WINDOW_LENGTH).astype(np.float32)


def generate_spectrograms(signal_batch):
    """
    Input:
        signal_batch: (batch, 5, 256) complex array

    Output:
        images: (batch, 229, 40), normalized to [0, 1]
    """

    # Central 150 slow-time samples
    signal_batch = signal_batch[
        :,
        :,
        CROP_START:CROP_END,
    ]

    # Shape: (batch, 5, 87, 64)
    frames = sliding_window_view(
        signal_batch,
        window_shape=WINDOW_LENGTH,
        axis=-1,
    )

    # Hop size = 2, producing 44 temporal frames
    frames = frames[:, :, ::HOP_LENGTH, :]

    # Hann window
    frames = frames * WINDOW

    # Complex STFT
    spectra = fft(
        frames,
        n=NFFT,
        axis=-1,
        workers=-1,
    )

    # Centre zero Doppler
    spectra = fftshift(spectra, axes=-1)

    # Non-coherent integration over five range cells
    power = np.sum(
        np.abs(spectra) ** 2,
        axis=1,
    )

    # Convert from (batch, time, frequency)
    # to (batch, frequency, time)
    power = np.transpose(power, (0, 2, 1))

    # Positive Doppler at top, negative Doppler at bottom
    power = power[:, ::-1, :]

    # Image-relative log-power normalization
    peak = np.max(
        power,
        axis=(1, 2),
        keepdims=True,
    )

    relative_power = power / np.maximum(
        peak,
        np.finfo(np.float32).tiny,
    )

    power_db = 10.0 * np.log10(
        np.maximum(relative_power, 1e-6)
    )

    power_db = np.clip(
        power_db,
        -DYNAMIC_RANGE_DB,
        0.0,
    )

    images = (
        power_db + DYNAMIC_RANGE_DB
    ) / DYNAMIC_RANGE_DB

    # Mandatory outer 5% border removal
    frequency_crop = round(images.shape[1] * 0.05)
    time_crop = round(images.shape[2] * 0.05)

    images = images[
        :,
        frequency_crop:-frequency_crop,
        time_crop:-time_crop,
    ]

    # Remove asymmetric Nyquist row so zero Doppler lies
    # exactly between the two central rows after resizing
    images = images[:, :-1, :]

    return images


def save_image(image, output_path):
    pixels = np.round(
        np.clip(image, 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    resized = Image.fromarray(pixels).resize(
        OUTPUT_SIZE,
        resample=Image.Resampling.BILINEAR,
    )

    resized.save(
        output_path,
        format="PNG",
        optimize=False,
        compress_level=1,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    data = np.load(DATASET, allow_pickle=True)

    total_created = 0
    total_existing = 0

    for measurement_id in range(data.shape[0]):
        signal = np.asarray(
            data[measurement_id, 1]
        )

        if signal.ndim != 2:
            raise ValueError(
                f"Measurement {measurement_id}: "
                f"invalid shape {signal.shape}"
            )

        if signal.shape[0] != 1280:
            raise ValueError(
                f"Measurement {measurement_id}: "
                f"expected 1280 rows, found {signal.shape[0]}"
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

        for segment_id in range(number_of_segments):
            output_path = (
                measurement_dir
                / f"segment_{segment_id:05d}.png"
            )

            if output_path.exists():
                total_existing += 1
            else:
                pending_segments.append(segment_id)

        for batch_start in range(
            0,
            len(pending_segments),
            BATCH_SIZE,
        ):
            batch_ids = pending_segments[
                batch_start:batch_start + BATCH_SIZE
            ]

            # (1280, batch) -> (batch, 5, 256)
            signal_batch = (
                signal[:, batch_ids]
                .T
                .reshape(len(batch_ids), 5, 256)
                .astype(np.complex64)
            )

            images = generate_spectrograms(
                signal_batch
            )

            for index, segment_id in enumerate(batch_ids):
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
            f"Measurement {measurement_id + 1:03d}/"
            f"{data.shape[0]} | "
            f"segments={number_of_segments} | "
            f"created={len(pending_segments)}"
        )

    print("\nCompleted")
    print("New images:", total_created)
    print("Existing images skipped:", total_existing)
    print("Total images:", total_created + total_existing)


if __name__ == "__main__":
    main()