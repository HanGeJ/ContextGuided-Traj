from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _read_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 4:
        raise ValueError(f"{path}: expected >=4 columns frame,ped,x,y; got {arr.shape}")
    out = arr[:, :4]
    if not np.isfinite(out).all():
        raise ValueError(f"{path}: contains non-finite values")
    return out


def _infer_frame_step(frames: np.ndarray) -> int:
    u = np.unique(frames.astype(np.int64))
    if len(u) < 2:
        return 1
    diff = np.diff(u)
    diff = diff[diff > 0]
    if len(diff) == 0:
        return 1
    counts = Counter(diff.tolist())
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def _split_contiguous(frames: np.ndarray, step: int) -> list[tuple[int, int]]:
    if len(frames) == 0:
        return []
    cuts = np.flatnonzero(np.diff(frames) != int(step)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(frames)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _frame_neighbors(frame_ids: np.ndarray, frame_xy: np.ndarray, topk: int):
    """Return a cache ped_id -> (relative[K,2], mask[K]) for one frame."""
    n = len(frame_ids)
    cache = {}
    if n == 0:
        return cache
    for i in range(n):
        rel = frame_xy - frame_xy[i]
        dist2 = (rel * rel).sum(axis=1)
        dist2[i] = np.inf
        order = np.argsort(dist2, kind="stable")[:topk]
        valid = np.isfinite(dist2[order])
        rr = np.zeros((topk, 2), dtype=np.float32)
        mm = np.zeros((topk,), dtype=np.float32)
        m = int(valid.sum())
        if m:
            rr[:m] = rel[order[valid]].astype(np.float32)
            mm[:m] = 1.0
        cache[int(frame_ids[i])] = (rr, mm)
    return cache


def build_clip(path: Path, clip_index: int, obs_len: int, pred_len: int,
               neighbor_topk: int, sample_stride: int, frame_step: int | None):
    raw = _read_txt(path)
    frames = raw[:, 0].astype(np.int64)
    peds = raw[:, 1].astype(np.int64)
    xy = raw[:, 2:4].astype(np.float32)

    step = int(frame_step) if frame_step is not None else _infer_frame_step(frames)
    frame_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fr in np.unique(frames):
        idx = np.flatnonzero(frames == fr)
        frame_map[int(fr)] = (peds[idx], xy[idx])

    neighbor_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for fr, (ids, coords) in frame_map.items():
        c = _frame_neighbors(ids, coords, neighbor_topk)
        for pid, val in c.items():
            neighbor_cache[(fr, pid)] = val

    by_ped = defaultdict(list)
    for fr, pid, pos in zip(frames, peds, xy):
        by_ped[int(pid)].append((int(fr), pos))

    total_len = int(obs_len + pred_len)
    obs_list, pred_list = [], []
    neigh_list, neigh_mask_list = [], []
    seq_list, ped_list, clip_list = [], [], []
    local_sid = 0

    for pid, rows in by_ped.items():
        rows.sort(key=lambda z: z[0])
        pf = np.asarray([r[0] for r in rows], dtype=np.int64)
        pp = np.stack([r[1] for r in rows]).astype(np.float32)
        for s, e in _split_contiguous(pf, step):
            run_len = e - s
            if run_len < total_len:
                continue
            for off in range(0, run_len - total_len + 1, max(1, int(sample_stride))):
                a = s + off
                b = a + total_len
                wf = pf[a:b]
                wp = pp[a:b]
                # Defensive exact-continuity check.
                if len(wf) > 1 and not np.all(np.diff(wf) == step):
                    continue
                neigh = np.zeros((obs_len, neighbor_topk, 2), dtype=np.float32)
                mask = np.zeros((obs_len, neighbor_topk), dtype=np.float32)
                for t, fr in enumerate(wf[:obs_len]):
                    val = neighbor_cache.get((int(fr), int(pid)))
                    if val is not None:
                        neigh[t], mask[t] = val

                obs_list.append(wp[:obs_len])
                pred_list.append(wp[obs_len:])
                neigh_list.append(neigh)
                neigh_mask_list.append(mask)
                # Keep clip identity in the million-prefix for compatibility
                # with the group-disjoint validation fallback.
                seq_list.append(int(clip_index) * 1_000_000 + int(local_sid))
                ped_list.append(int(pid))
                clip_list.append(int(clip_index))
                local_sid += 1

    if not obs_list:
        return None, {"clip": path.stem, "frame_step": step, "windows": 0,
                      "rows": int(len(raw)), "pedestrians": int(len(by_ped))}

    data = {
        "obs": np.stack(obs_list).astype(np.float32),
        "pred": np.stack(pred_list).astype(np.float32),
        "neighbor_rel": np.stack(neigh_list).astype(np.float32),
        "neighbor_mask": np.stack(neigh_mask_list).astype(np.float32),
        "seq_id": np.asarray(seq_list, dtype=np.int64),
        "ped_id": np.asarray(ped_list, dtype=np.int64),
        "clip_index": np.asarray(clip_list, dtype=np.int64),
        "dt": np.full(len(obs_list), 0.4, dtype=np.float32),
    }
    stat = {
        "clip": path.stem,
        "frame_step": step,
        "windows": int(len(obs_list)),
        "rows": int(len(raw)),
        "pedestrians": int(len(by_ped)),
        "neighbor_valid_ratio": float(data["neighbor_mask"].mean()),
    }
    return data, stat


def combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise RuntimeError("No valid SDD windows were extracted.")
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def save_npz(path: Path, data: dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def main():
    ap = argparse.ArgumentParser(description="Prepare the public TUTR SDD split for TMM trajectory experiments.")
    ap.add_argument("--tutr_root", required=True,
                    help="Path to cloned lssiair/TUTR repository")
    ap.add_argument("--out_dir", default="data/sdd_tutr")
    ap.add_argument("--obs_len", type=int, default=8)
    ap.add_argument("--pred_len", type=int, default=12)
    ap.add_argument("--neighbor_topk", type=int, default=8)
    ap.add_argument("--sample_stride", type=int, default=1)
    ap.add_argument("--frame_step", type=str, default="auto")
    args = ap.parse_args()

    root = Path(args.tutr_root).resolve()
    sdd_root = root / "data" / "sdd"
    train_dir, test_dir = sdd_root / "train", sdd_root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected TUTR SDD data under {sdd_root}/{{train,test}}. "
            "Clone https://github.com/lssiair/TUTR first."
        )

    frame_step = None if str(args.frame_step).lower() == "auto" else int(args.frame_step)
    out_fold = Path(args.out_dir) / "sdd"
    manifest = {
        "source": "lssiair/TUTR data/sdd",
        "source_url": "https://github.com/lssiair/TUTR",
        "protocol": "official TUTR/SocialVAE SDD train-test recording split; internal group-disjoint validation from train only",
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "dt_seconds": 0.4,
        "neighbor_topk": int(args.neighbor_topk),
        "sample_stride": int(args.sample_stride),
        "splits": {},
        "clip_map": {},
    }

    clip_counter = 0
    for split, directory in [("train", train_dir), ("test", test_dir)]:
        parts, stats = [], []
        for path in sorted(directory.glob("*.txt")):
            clip_counter += 1
            manifest["clip_map"][str(clip_counter)] = path.stem
            data, stat = build_clip(
                path, clip_counter, args.obs_len, args.pred_len,
                args.neighbor_topk, args.sample_stride, frame_step,
            )
            stat["clip_index"] = int(clip_counter)
            stats.append(stat)
            if data is not None:
                parts.append(data)
        merged = combine(parts)
        save_npz(out_fold / f"{split}.npz", merged)
        manifest["splits"][split] = {
            "clips": stats,
            "num_clips": int(len(stats)),
            "num_windows": int(len(merged["obs"])),
            "neighbor_valid_ratio": float(merged["neighbor_mask"].mean()),
            "position_abs_max": float(np.abs(np.concatenate([merged["obs"], merged["pred"]], axis=1)).max()),
        }
        print(
            f"{split}: clips={len(stats)} windows={len(merged['obs'])} "
            f"obs={merged['obs'].shape} pred={merged['pred'].shape} "
            f"neighbor_valid={merged['neighbor_mask'].mean():.4f}"
        )

    (out_fold / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved SDD fold to: {out_fold.resolve()}")


if __name__ == "__main__":
    main()
