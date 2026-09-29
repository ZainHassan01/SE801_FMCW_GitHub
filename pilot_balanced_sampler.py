import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from pilot_fold_loader import ROOT, SpectrogramDataset


class BalancedBatchSampler(Sampler):
    def __init__(self, rows, batch_size=16, batches_per_epoch=4, seed=2026):
        if batch_size % 4:
            raise ValueError("Batch size must be divisible by four")

        self.per_class = batch_size // 4
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        self.groups = {}
        self.labels = rows["class ID"].to_numpy()

        for class_id, class_rows in rows.groupby("class ID"):
            # .groups preserves indices into the full training dataset.
            self.groups[int(class_id)] = {
                int(parent_id): np.asarray(indices, dtype=int)
                for parent_id, indices in class_rows.groupby(
                    "original-image identifier"
                ).groups.items()
            }

        if set(self.groups) != {0, 1, 2, 3}:
            raise ValueError("Training split lacks a class")

        for class_id, parents in self.groups.items():
            if len(parents) < self.per_class:
                raise ValueError(f"Too few parents in class {class_id}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        for _ in range(self.batches_per_epoch):
            batch = []

            for class_id in range(4):
                parents = self.groups[class_id]
                selected = rng.choice(
                    list(parents), size=self.per_class, replace=False
                )

                for parent_id in selected:
                    index = int(rng.choice(parents[int(parent_id)]))
                    if int(self.labels[index]) != class_id:
                        raise RuntimeError("Sampler index maps to wrong class")
                    batch.append(index)

            rng.shuffle(batch)
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    args = parser.parse_args()

    folds = pd.read_csv(ROOT / "manifests_224x48/fold_manifest.csv")
    outer = folds[folds["outer fold"] == args.outer]
    train = outer[
        outer["inner fold"].notna()
        & (outer["inner fold"] != args.inner)
    ].reset_index(drop=True)

    with (ROOT / "artifacts_224x48/common/fold_normalization.json").open() as f:
        all_stats = json.load(f)

    stats = all_stats["inner_training"][
        f"outer_{args.outer}_inner_{args.inner}"
    ]
    if stats["images"] != len(train):
        raise ValueError("Normalization statistics do not match training fold")

    dataset = SpectrogramDataset(
        train, stats["mean"], stats["standard_deviation"]
    )
    sampler = BalancedBatchSampler(train)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)

    for batch_number, (images, labels, parent_ids) in enumerate(loader, 1):
        counts = torch.bincount(labels, minlength=4).tolist()
        unique_parents = len(set(parent_ids.tolist()))

        print(
            f"Batch {batch_number}: classes={counts}, "
            f"unique_parents={unique_parents}"
        )

        if counts != [4, 4, 4, 4] or unique_parents != 16:
            raise RuntimeError("Batch balance check failed")

        images = images.to("cuda")
        if not torch.isfinite(images).all():
            raise RuntimeError("Nonfinite image values")

    print("BALANCED_SAMPLER_PILOT_PASS")


if __name__ == "__main__":
    main()