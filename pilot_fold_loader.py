import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/data/szain/SE801_FMCW")


class SpectrogramDataset(Dataset):
    def __init__(self, rows, mean, std):
        self.rows = rows.reset_index(drop=True)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(ROOT / row["filename"]) as image:
            if image.size != (48, 224) or image.mode != "L":
                raise ValueError(f"Invalid image: {row['filename']}")
            pixels = np.asarray(image, dtype=np.float32) / 255.0

        pixels = (pixels - self.mean) / self.std
        x = torch.from_numpy(pixels.copy()).unsqueeze(0)
        return x, int(row["class ID"]), int(row["original-image identifier"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    args = parser.parse_args()

    if args.outer not in range(1, 6) or args.inner not in range(1, 6):
        raise ValueError("Fold numbers must be 1–5")

    folds = pd.read_csv(ROOT / "manifests_224x48/fold_manifest.csv")
    rows = folds.loc[folds["outer fold"] == args.outer]

    test = rows.loc[rows["inner fold"].isna()]
    validation = rows.loc[rows["inner fold"] == args.inner]
    train = rows.loc[
        rows["inner fold"].notna() & (rows["inner fold"] != args.inner)
    ]

    splits = {"train": train, "validation": validation, "test": test}
    for name, split in splits.items():
        if not split["filename"].is_unique:
            raise ValueError(f"Duplicate filename in {name}")
        if set(split["class ID"]) != {0, 1, 2, 3}:
            raise ValueError(f"Missing class in {name}")

    groups = {
        name: set(split["original-image identifier"])
        for name, split in splits.items()
    }
    if (groups["train"] & groups["validation"]
            or groups["train"] & groups["test"]
            or groups["validation"] & groups["test"]):
        raise ValueError("Parent measurement crosses partitions")

    with (ROOT / "artifacts_224x48/common/fold_normalization.json").open() as f:
        normalization = json.load(f)

    stats = normalization["inner_training"][
        f"outer_{args.outer}_inner_{args.inner}"
    ]
    if stats["images"] != len(train) or stats["parent_measurements"] != len(groups["train"]):
        raise ValueError("Normalization statistics belong to another split")

    for name, split in splits.items():
        print(f"{name}: {len(split)} images, {len(groups[name])} parents")

    dataset = SpectrogramDataset(
        train, stats["mean"], stats["standard_deviation"]
    )
    loader = DataLoader(dataset, batch_size=16, num_workers=2)
    images, labels, parent_ids = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)

    print("Batch shape:", tuple(images.shape))
    print("Device:", device)
    print("Batch labels:", labels.tolist())
    print("Batch parent IDs:", parent_ids.tolist())
    print("FOLD_LOADER_PILOT_PASS")


if __name__ == "__main__":
    main()