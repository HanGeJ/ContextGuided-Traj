from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import TrajectoryWindowDataset, _group_disjoint_split, _limited_indices, resolve_fold_paths


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


class VisualTrajectoryWindowDataset(Dataset):
    """Trajectory windows augmented with a pre-cached local RGB scene crop."""

    def __init__(
        self,
        npz_path: str | Path,
        crop_path: str | Path,
        indices: Optional[np.ndarray] = None,
        relative_coords: bool = True,
        neighbor_topk_default: int = 8,
        normalize_imagenet: bool = True,
    ):
        npz_path = Path(npz_path)
        crop_path = Path(crop_path)
        with np.load(npz_path, allow_pickle=False) as d:
            n_all = len(d["obs"])
        if indices is None:
            indices = np.arange(n_all, dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)
        self.indices = indices
        self.base = TrajectoryWindowDataset(
            npz_path, indices=indices, relative_coords=relative_coords,
            neighbor_topk_default=neighbor_topk_default,
        )
        if not crop_path.exists():
            raise FileNotFoundError(f"Scene crop cache not found: {crop_path}")
        self.crops = np.load(crop_path, mmap_mode="r")
        if len(self.crops) != n_all:
            raise ValueError(
                f"Crop/sample mismatch for {npz_path}: crops={len(self.crops)} samples={n_all}"
            )
        if self.crops.ndim != 4 or self.crops.shape[1] != 3:
            raise ValueError(f"Expected crop array [N,3,H,W], got {self.crops.shape}")
        self.normalize_imagenet = bool(normalize_imagenet)

        # Deterministic hard-negative scene pairing.  Earlier Stage-3/4 audits
        # used torch.roll within a sequential validation batch, which can swap
        # two adjacent windows from the same recording and therefore does not
        # reliably break scene correspondence.  Here every local sample is
        # paired with a crop from a different trajectory group, preferring a
        # different semantic scene when that metadata is available.
        clip = np.asarray(self.base.clip_index, dtype=np.int64)
        seq_group = np.asarray(self.base.seq_id, dtype=np.int64) // 1_000_000
        group = np.where(clip >= 0, clip, seq_group)
        scene = np.asarray(self.base.scene_index, dtype=np.int64)
        self.negative_local_index = self._build_negative_index(group, scene)

    @staticmethod
    def _build_negative_index(group: np.ndarray, scene: np.ndarray) -> np.ndarray:
        n = len(group)
        out = np.empty(n, dtype=np.int64)
        all_idx = np.arange(n, dtype=np.int64)
        for i in range(n):
            # First preference: a different semantic scene.  This is available
            # for SDD and is a stronger counterfactual than an adjacent window.
            if scene[i] >= 0:
                cand = all_idx[(scene >= 0) & (scene != scene[i])]
            else:
                cand = np.empty(0, dtype=np.int64)
            # Fallback: a different clip/raw-sequence group.
            if len(cand) == 0:
                cand = all_idx[group != group[i]]
            # Last-resort deterministic rotation for degenerate one-group data.
            if len(cand) == 0:
                out[i] = (i + max(1, n // 2)) % max(1, n)
            else:
                out[i] = int(cand[i % len(cand)])
        return out

    def _crop_tensor(self, raw_index: int) -> torch.Tensor:
        crop_np = np.array(self.crops[int(raw_index)], copy=True)
        crop = torch.from_numpy(crop_np).to(torch.float32).div_(255.0)
        if self.normalize_imagenet:
            crop = (crop - _IMAGENET_MEAN) / _IMAGENET_STD
        return crop

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        raw_index = int(self.indices[idx])
        neg_local = int(self.negative_local_index[idx])
        neg_raw = int(self.indices[neg_local])
        item["scene_crop"] = self._crop_tensor(raw_index)
        item["scene_crop_negative"] = self._crop_tensor(neg_raw)
        item["scene_negative_local_index"] = neg_local
        item["scene_negative_raw_index"] = neg_raw
        return item


def build_visual_dataloaders(
    data_root: str | Path,
    test_set: str,
    crop_tag: str,
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    val_source: str = "train_group",
    relative_coords: bool = True,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    pin_memory: bool = True,
):
    fold = resolve_fold_paths(data_root, test_set)
    val_source = val_source.lower()
    if val_source not in ("train_group", "train", "test", "test_split"):
        raise ValueError("Unsupported val_source")
    root = Path(data_root) / str(test_set)
    crop_dir = root / f"scene_crops_{crop_tag}"
    train_crop = crop_dir / "train.npy"
    test_crop = crop_dir / "test.npy"

    with np.load(fold.train_path, allow_pickle=False) as d:
        n_train_all = len(d["obs"])
    with np.load(fold.test_path, allow_pickle=False) as d:
        n_test_all = len(d["obs"])

    rng = np.random.default_rng(split_seed)
    all_train = np.arange(n_train_all, dtype=np.int64)
    all_test = np.arange(n_test_all, dtype=np.int64)

    if val_source == "train_group":
        split = _group_disjoint_split(fold.train_path, val_ratio, split_seed)
        if split is None:
            rng.shuffle(all_train)
            n_val = max(1, int(round(n_train_all * val_ratio)))
            val_idx, train_idx = np.sort(all_train[:n_val]), np.sort(all_train[n_val:])
        else:
            train_idx, val_idx = split
        test_idx = all_test
    elif val_source == "train":
        shuffled = all_train.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(n_train_all * val_ratio)))
        val_idx, train_idx = np.sort(shuffled[:n_val]), np.sort(shuffled[n_val:])
        test_idx = all_test
    elif val_source == "test_split":
        shuffled = all_test.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(n_test_all * val_ratio)))
        val_idx = np.sort(shuffled[:n_val])
        test_idx = np.sort(shuffled[n_val:])
        train_idx = all_train
    else:  # legacy: use full test as validation
        train_idx, val_idx, test_idx = all_train, all_test, all_test

    train_idx = train_idx[_limited_indices(len(train_idx), max_train_samples, split_seed)] if max_train_samples else train_idx
    val_idx = val_idx[_limited_indices(len(val_idx), max_val_samples, split_seed + 1)] if max_val_samples else val_idx
    test_idx = test_idx[_limited_indices(len(test_idx), max_test_samples, split_seed + 2)] if max_test_samples else test_idx

    train_ds = VisualTrajectoryWindowDataset(fold.train_path, train_crop, train_idx, relative_coords)
    val_npz = fold.train_path if val_source in ("train_group", "train") else fold.test_path
    val_crop = train_crop if val_source in ("train_group", "train") else test_crop
    val_ds = VisualTrajectoryWindowDataset(val_npz, val_crop, val_idx, relative_coords)
    test_ds = VisualTrajectoryWindowDataset(fold.test_path, test_crop, test_idx, relative_coords)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        common.update(dict(persistent_workers=True, prefetch_factor=2))
    return {
        "train": DataLoader(train_ds, shuffle=True, drop_last=False, **common),
        "val": DataLoader(val_ds, shuffle=False, drop_last=False, **common),
        "test": DataLoader(test_ds, shuffle=False, drop_last=False, **common),
    }
