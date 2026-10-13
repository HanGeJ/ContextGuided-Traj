from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Retained for backward compatibility with scripts that import this symbol.
VALID_TEST_SETS = ("eth", "hotel", "univ", "zara1", "zara2")


def list_available_folds(data_root: str | Path) -> tuple[str, ...]:
    """Return directories that contain both train.npz and test.npz."""
    root = Path(data_root)
    if not root.exists():
        return ()
    folds = [p.name for p in root.iterdir()
             if p.is_dir() and (p / "train.npz").exists() and (p / "test.npz").exists()]
    return tuple(sorted(folds))


def _take(arrays: np.lib.npyio.NpzFile, key: str, indices: np.ndarray,
          default: np.ndarray) -> np.ndarray:
    if key not in arrays:
        return default
    return np.asarray(arrays[key])[indices]


class TrajectoryWindowDataset(Dataset):
    """Trajectory windows with optional ego-vehicle and safety metadata.

    Required NPZ fields:
        obs:  [N, T_obs, 2] absolute target positions
        pred: [N, T_pred, 2] absolute target positions

    Optional social-context and mixed-traffic fields are loaded when present.

    Social-context fields:
        neighbor_rel:  [N, T_obs, K_neighbor, 2]
            Relative neighbor positions with respect to the target pedestrian
            at the same observation step.
        neighbor_mask: [N, T_obs, K_neighbor]
            Binary validity mask for neighbor_rel.

    For datasets without social or ego context, zero-valued placeholders are
    generated so that training/evaluation code can use one common batch
    interface across datasets.
    """

    def __init__(self, npz_path: str | Path, indices: Optional[np.ndarray] = None,
                 relative_coords: bool = True, neighbor_topk_default: int = 4):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as arrays:
            obs_all = np.asarray(arrays["obs"], dtype=np.float32)
            pred_all = np.asarray(arrays["pred"], dtype=np.float32)
            n_all = len(obs_all)
            indices = (np.arange(n_all, dtype=np.int64) if indices is None
                       else np.asarray(indices, dtype=np.int64))

            self.obs_abs = obs_all[indices]
            self.pred_abs = pred_all[indices]
            n = len(indices)
            obs_len = self.obs_abs.shape[1]
            pred_len = self.pred_abs.shape[1]

            self.seq_id = _take(
                arrays, "seq_id", indices, np.arange(n_all, dtype=np.int64)[indices]
            ).astype(np.int64, copy=False)
            self.ped_id = _take(
                arrays, "ped_id", indices, np.full(n, -1, dtype=np.int64)
            ).astype(np.int64, copy=False)
            self.ego_id = _take(
                arrays, "ego_id", indices, np.full(n, -1, dtype=np.int64)
            ).astype(np.int64, copy=False)
            self.clip_index = _take(
                arrays, "clip_index", indices, np.full(n, -1, dtype=np.int64)
            ).astype(np.int64, copy=False)
            self.scenario_type = _take(
                arrays, "scenario_type", indices, np.full(n, -1, dtype=np.int64)
            ).astype(np.int64, copy=False)
            self.scene_index = _take(
                arrays, "scene_index", indices, np.full(n, -1, dtype=np.int64)
            ).astype(np.int64, copy=False)
            self.video_index = _take(
                arrays, "video_index", indices, np.full(n, -1, dtype=np.int64)
            ).astype(np.int64, copy=False)

            # --------------------------------------------------------------
            # Optional social-neighbor context.
            #
            # ETH/UCY preprocessing stores neighbor_rel as the displacement
            # from the target pedestrian to each neighbor at the same time
            # step. It is therefore already translation invariant and should
            # not be re-centered by the target's final observed position.
            # --------------------------------------------------------------
            has_neighbor_rel = "neighbor_rel" in arrays
            has_neighbor_mask = "neighbor_mask" in arrays
            if has_neighbor_rel != has_neighbor_mask:
                raise ValueError(
                    f"{npz_path}: neighbor_rel and neighbor_mask must either "
                    "both be present or both be absent."
                )

            if has_neighbor_rel:
                neighbor_rel_all = np.asarray(
                    arrays["neighbor_rel"], dtype=np.float32
                )
                neighbor_mask_all = np.asarray(
                    arrays["neighbor_mask"], dtype=np.float32
                )

                if (
                    neighbor_rel_all.ndim != 4
                    or neighbor_rel_all.shape[0] != n_all
                    or neighbor_rel_all.shape[1] != obs_len
                    or neighbor_rel_all.shape[-1] != 2
                ):
                    raise ValueError(
                        f"{npz_path}: invalid neighbor_rel shape "
                        f"{neighbor_rel_all.shape}; expected "
                        f"[N, {obs_len}, K_neighbor, 2]."
                    )

                expected_mask_shape = neighbor_rel_all.shape[:3]
                if neighbor_mask_all.shape != expected_mask_shape:
                    raise ValueError(
                        f"{npz_path}: invalid neighbor_mask shape "
                        f"{neighbor_mask_all.shape}; expected "
                        f"{expected_mask_shape}."
                    )

                self.neighbor_rel = neighbor_rel_all[indices].astype(
                    np.float32, copy=False
                )
                self.neighbor_mask = (
                    neighbor_mask_all[indices] > 0.5
                ).astype(np.float32, copy=False)
            else:
                k_neighbor = max(1, int(neighbor_topk_default))
                self.neighbor_rel = np.zeros(
                    (n, obs_len, k_neighbor, 2), dtype=np.float32
                )
                self.neighbor_mask = np.zeros(
                    (n, obs_len, k_neighbor), dtype=np.float32
                )

            # Force padded neighbors to exactly zero. This prevents stale or
            # malformed padded values from leaking into future social encoders.
            self.neighbor_rel = (
                self.neighbor_rel * self.neighbor_mask[..., None]
            ).astype(np.float32, copy=False)

            self.ego_obs_abs = _take(
                arrays, "ego_obs_abs", indices,
                np.zeros((n, obs_len, 2), dtype=np.float32)
            ).astype(np.float32, copy=False)
            self.ego_pred_abs = _take(
                arrays, "ego_pred_abs", indices,
                np.zeros((n, pred_len, 2), dtype=np.float32)
            ).astype(np.float32, copy=False)
            self.ego_valid = _take(
                arrays, "ego_valid", indices, np.zeros(n, dtype=np.float32)
            ).astype(np.float32, copy=False)
            self.ego_speed_mps = _take(
                arrays, "ego_speed_mps", indices, np.zeros(n, dtype=np.float32)
            ).astype(np.float32, copy=False)
            self.gt_conflict = _take(
                arrays, "gt_conflict", indices, np.zeros(n, dtype=np.float32)
            ).astype(np.float32, copy=False)
            self.gt_conflict_time_s = _take(
                arrays, "gt_conflict_time_s", indices, np.full(n, -1.0, dtype=np.float32)
            ).astype(np.float32, copy=False)

            if "dt" in arrays:
                dt_raw = np.asarray(arrays["dt"])
                if dt_raw.ndim == 0:
                    self.dt = np.full(n, float(dt_raw), dtype=np.float32)
                else:
                    self.dt = dt_raw[indices].astype(np.float32, copy=False)
            else:
                self.dt = np.full(n, 0.4, dtype=np.float32)

        self.relative_coords = bool(relative_coords)
        if self.relative_coords:
            self.origin = self.obs_abs[:, -1, :].copy()
            self.obs = self.obs_abs - self.origin[:, None, :]
            self.pred = self.pred_abs - self.origin[:, None, :]
            # Vehicle coordinates use the target pedestrian's last observed
            # position as the common origin. This preserves target-vehicle range.
            self.ego_obs_rel = self.ego_obs_abs - self.origin[:, None, :]
            self.ego_pred_rel = self.ego_pred_abs - self.origin[:, None, :]
        else:
            self.origin = np.zeros((len(self.obs_abs), 2), dtype=np.float32)
            self.obs = self.obs_abs
            self.pred = self.pred_abs
            self.ego_obs_rel = self.ego_obs_abs
            self.ego_pred_rel = self.ego_pred_abs

        # Ensure invalid ego placeholders remain exactly zero after subtraction.
        invalid = self.ego_valid < 0.5
        self.ego_obs_rel[invalid] = 0.0
        self.ego_pred_rel[invalid] = 0.0

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "pred": torch.from_numpy(self.pred[idx]),
            "origin": torch.from_numpy(self.origin[idx]),
            "obs_abs": torch.from_numpy(self.obs_abs[idx]),
            "pred_abs": torch.from_numpy(self.pred_abs[idx]),
            "neighbor_rel": torch.from_numpy(self.neighbor_rel[idx]),
            "neighbor_mask": torch.from_numpy(self.neighbor_mask[idx]),
            "ego_obs_abs": torch.from_numpy(self.ego_obs_abs[idx]),
            "ego_pred_abs": torch.from_numpy(self.ego_pred_abs[idx]),
            "ego_obs_rel": torch.from_numpy(self.ego_obs_rel[idx]),
            "ego_pred_rel": torch.from_numpy(self.ego_pred_rel[idx]),
            "ego_valid": torch.tensor(self.ego_valid[idx], dtype=torch.float32),
            "ego_speed_mps": torch.tensor(self.ego_speed_mps[idx], dtype=torch.float32),
            "gt_conflict": torch.tensor(self.gt_conflict[idx], dtype=torch.float32),
            "gt_conflict_time_s": torch.tensor(self.gt_conflict_time_s[idx], dtype=torch.float32),
            "dt": torch.tensor(self.dt[idx], dtype=torch.float32),
            "seq_id": int(self.seq_id[idx]),
            "ped_id": int(self.ped_id[idx]),
            "ego_id": int(self.ego_id[idx]),
            "clip_index": int(self.clip_index[idx]),
            "scenario_type": int(self.scenario_type[idx]),
            "scene_index": int(self.scene_index[idx]),
            "video_index": int(self.video_index[idx]),
        }


@dataclass
class FoldDatasets:
    train_path: Path
    test_path: Path


def resolve_fold_paths(data_root: str | Path, test_set: str) -> FoldDatasets:
    data_root = Path(data_root)
    fold_dir = data_root / str(test_set)
    train_path = fold_dir / "train.npz"
    test_path = fold_dir / "test.npz"
    if not train_path.exists() or not test_path.exists():
        available = list_available_folds(data_root)
        raise FileNotFoundError(
            f"Expected train.npz and test.npz at {fold_dir}. "
            f"Available folds: {available or 'none'}"
        )
    return FoldDatasets(train_path=train_path, test_path=test_path)


def _limited_indices(n: int, limit: Optional[int], seed: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    if limit is None or limit <= 0 or limit >= n:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=limit, replace=False))


def _group_labels_for_validation(npz_path: Path) -> np.ndarray | None:
    """Return clip/recording groups suitable for validation splitting.

    Mixed-traffic files expose ``clip_index`` directly. Older ETH/UCY files
    encode the source raw sequence in the million-prefix of ``seq_id``. A
    group-disjoint split prevents heavily overlapping windows from the same
    clip or recording from appearing in both training and validation.
    """
    with np.load(npz_path, allow_pickle=False) as arrays:
        if "clip_index" in arrays:
            labels = np.asarray(arrays["clip_index"], dtype=np.int64)
            valid = labels >= 0
            if valid.all() and np.unique(labels).size >= 2:
                return labels
        if "seq_id" in arrays:
            seq_id = np.asarray(arrays["seq_id"], dtype=np.int64)
            labels = seq_id // 1_000_000
            if np.unique(labels).size >= 2:
                return labels
    return None


def _group_disjoint_split(npz_path: Path, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    labels = _group_labels_for_validation(npz_path)
    if labels is None:
        return None
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        return None

    target = max(1, int(round(len(labels) * max(float(val_ratio), 0.0))))
    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(unique))
    remaining = list(range(len(unique)))
    selected: list[int] = []
    selected_count = 0

    # Greedily choose whole groups while each addition improves proximity to
    # the requested validation size. Random tie-breaking keeps the split
    # reproducible under ``seed`` without privileging low-numbered clips.
    while remaining and len(selected) < len(unique) - 1:
        best = min(
            remaining,
            key=lambda i: (
                abs(target - (selected_count + int(counts[i]))),
                tie_break[i],
            ),
        )
        current_gap = abs(target - selected_count)
        proposed_gap = abs(target - (selected_count + int(counts[best])))
        if selected and proposed_gap >= current_gap:
            break
        selected.append(best)
        remaining.remove(best)
        selected_count += int(counts[best])
        if selected_count >= target:
            break

    if not selected:
        selected = [min(range(len(unique)), key=lambda i: (abs(target - int(counts[i])), tie_break[i]))]
    if len(selected) == len(unique):
        selected = selected[:-1]
    val_groups = unique[np.asarray(selected, dtype=np.int64)]
    val_mask = np.isin(labels, val_groups)
    val_idx = np.flatnonzero(val_mask).astype(np.int64)
    train_idx = np.flatnonzero(~val_mask).astype(np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        return None
    return train_idx, val_idx


def build_dataloaders(data_root: str | Path, test_set: str, batch_size: int = 128,
                      num_workers: int = 0, val_ratio: float = 0.1,
                      split_seed: int = 42, val_source: str = "train_group",
                      relative_coords: bool = True,
                      max_train_samples: Optional[int] = None,
                      max_val_samples: Optional[int] = None,
                      max_test_samples: Optional[int] = None,
                      pin_memory: bool = False) -> Dict[str, DataLoader]:
    """Build loaders without using the held-out test fold for model selection.

    The default ``val_source='train_group'`` reserves complete clips or raw
    recording groups from the training fold, preventing overlap between nearby
    trajectory windows. ``val_source='train'`` performs a sample-disjoint
    random split. Test-based options remain available only for legacy runs.
    """
    fold = resolve_fold_paths(data_root, test_set)
    val_source = val_source.lower()
    if val_source not in ("train_group", "train", "test", "test_split"):
        raise ValueError(
            "val_source must be 'train_group', 'train', 'test', or 'test_split'."
        )

    rng = np.random.default_rng(split_seed)
    with np.load(fold.train_path, allow_pickle=False) as a:
        n_train_all = len(a["obs"])
    with np.load(fold.test_path, allow_pickle=False) as a:
        n_test_all = len(a["obs"])

    if val_source in {"train_group", "train"}:
        if n_train_all < 2:
            raise ValueError(f"Training fold has too few samples: {n_train_all}")
        grouped = (
            _group_disjoint_split(fold.train_path, val_ratio, split_seed)
            if val_source == "train_group" else None
        )
        if grouped is None:
            if val_source == "train_group":
                warnings.warn(
                    "No usable clip/recording groups were found; falling back to a "
                    "sample-disjoint training split.",
                    RuntimeWarning,
                )
            shuffled = np.arange(n_train_all, dtype=np.int64)
            rng.shuffle(shuffled)
            val_size = max(1, int(round(n_train_all * val_ratio))) if val_ratio > 0 else 1
            val_size = min(val_size, n_train_all - 1)
            val_idx = shuffled[:val_size]
            train_idx = shuffled[val_size:]
        else:
            train_idx, val_idx = grouped
        if max_train_samples and max_train_samples < len(train_idx):
            train_idx = np.sort(rng.choice(train_idx, max_train_samples, replace=False))
        if max_val_samples and max_val_samples < len(val_idx):
            val_idx = np.sort(rng.choice(val_idx, max_val_samples, replace=False))
        test_idx = _limited_indices(n_test_all, max_test_samples, split_seed + 2)
        train_ds = TrajectoryWindowDataset(fold.train_path, train_idx, relative_coords)
        val_ds = TrajectoryWindowDataset(fold.train_path, val_idx, relative_coords)
        test_ds = TrajectoryWindowDataset(fold.test_path, test_idx, relative_coords)
    elif val_source == "test_split":
        warnings.warn(
            "Validation is drawn from the held-out test fold. Use only for legacy reproduction.",
            RuntimeWarning,
        )
        train_idx = _limited_indices(n_train_all, max_train_samples, split_seed)
        shuffled = np.arange(n_test_all, dtype=np.int64)
        rng.shuffle(shuffled)
        val_size = max(1, int(round(n_test_all * val_ratio))) if val_ratio > 0 else 1
        val_size = min(val_size, n_test_all - 1)
        val_idx, test_idx = shuffled[:val_size], shuffled[val_size:]
        if max_val_samples and max_val_samples < len(val_idx):
            val_idx = np.sort(rng.choice(val_idx, max_val_samples, replace=False))
        if max_test_samples and max_test_samples < len(test_idx):
            test_idx = np.sort(rng.choice(test_idx, max_test_samples, replace=False))
        train_ds = TrajectoryWindowDataset(fold.train_path, train_idx, relative_coords)
        val_ds = TrajectoryWindowDataset(fold.test_path, val_idx, relative_coords)
        test_ds = TrajectoryWindowDataset(fold.test_path, test_idx, relative_coords)
    else:
        warnings.warn(
            "val_source='test' reuses the test set for checkpoint selection and causes data leakage.",
            RuntimeWarning,
        )
        train_idx = _limited_indices(n_train_all, max_train_samples, split_seed)
        test_idx = _limited_indices(n_test_all, max_test_samples, split_seed + 2)
        train_ds = TrajectoryWindowDataset(fold.train_path, train_idx, relative_coords)
        shared = TrajectoryWindowDataset(fold.test_path, test_idx, relative_coords)
        val_ds = shared
        test_ds = shared

    common = dict(num_workers=num_workers, pin_memory=pin_memory)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
    }
