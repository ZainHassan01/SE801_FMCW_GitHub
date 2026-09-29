"""Parent-safe fold loader for native STFT PNGs with common 224x224 preprocessing.

Compatible with prior callers: ROOT and SpectrogramDataset(rows, mean, std).
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.preprocessing import preprocess_image


ROOT = Path(os.environ.get("SE801_FMCW_ROOT", "/data/szain/SE801_FMCW"))
if not ROOT.is_dir():
    ROOT = Path(__file__).resolve().parent


def native_path(filename: str) -> Path:
    """Map an existing fold-manifest filename to the same native segment ID."""
    name = str(filename).replace("\\", "/")
    parts = Path(name).parts
    measurement = next((part for part in parts if part.startswith("measurement_") and part[12:].isdigit()), None)
    segment = next((part for part in parts if part.startswith("segment_") and part[8:].endswith(".png")
                    and part[8:-4].isdigit()), None)
    if measurement is None or segment is None:
        raise ValueError(f"Cannot map manifest filename to native PNG: {filename}")
    return ROOT / "spectrograms_native_255x44" / measurement / segment


class SpectrogramDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, mean: float, std: float):
        self.rows = rows.reset_index(drop=True)
        self.mean = float(mean)
        self.std = float(std)
        if self.std <= 0:
            raise ValueError("Invalid training-fold standard deviation")
        needed = {"filename", "class ID", "original-image identifier"}
        if not needed.issubset(self.rows.columns):
            raise ValueError(f"Missing columns: {needed - set(self.rows.columns)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[int(index)]
        image = preprocess_image(native_path(row["filename"]), mean=self.mean, std=self.std)
        return torch.from_numpy(image), int(row["class ID"]), int(row["original-image identifier"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--stats", type=Path, default=ROOT / "artifacts_224x224/common/fold_normalization.json")
    args = parser.parse_args()
    if not (1 <= args.outer <= 5 and 1 <= args.inner <= 5):
        parser.error("fold IDs must be 1..5")

    frame = pd.read_csv(ROOT / "manifests_224x48/fold_manifest.csv")
    outer = frame.loc[frame["outer fold"] == args.outer]
    train = outer.loc[outer["inner fold"].notna() & (outer["inner fold"] != args.inner)].reset_index(drop=True)
    val = outer.loc[outer["inner fold"] == args.inner].reset_index(drop=True)
    test = outer.loc[outer["inner fold"].isna()].reset_index(drop=True)
    parents = [set(part["original-image identifier"]) for part in (train, val, test)]
    if any(parents[i] & parents[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Parent measurement leakage")
    with args.stats.open() as f:
        stats = json.load(f)["inner_training"][f"outer_{args.outer}_inner_{args.inner}"]
    if stats["images"] != len(train):
        raise RuntimeError("Stats do not correspond to this training partition")
    dataset = SpectrogramDataset(train, stats["mean"], stats["standard_deviation"])
    batch = next(iter(DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)))
    print(f"train={len(train)} val={len(val)} outer_test={len(test)}")
    print(f"parent_groups={[len(groups) for groups in parents]}")
    print(f"batch_shape={tuple(batch[0].shape)}")
    if tuple(batch[0].shape) != (16, 1, 224, 224):
        raise RuntimeError("Unexpected batch geometry")
    print("FOLD_LOADER_224x224_PASS")


if __name__ == "__main__":
    main()
