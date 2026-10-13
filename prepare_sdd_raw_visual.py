from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SCENES = ("bookstore", "coupa", "deathCircle", "gates", "hyang", "little", "nexus", "quad")


def _clip_from_stem(stem: str) -> tuple[str, int]:
    if "_" not in stem:
        raise ValueError(f"Cannot parse SDD clip name: {stem}")
    scene, video = stem.rsplit("_", 1)
    return scene, int(video)


def _load_split_clips(tutr_root: Path) -> dict[str, list[tuple[str, int, str]]]:
    sdd = tutr_root / "data" / "sdd"
    out: dict[str, list[tuple[str, int, str]]] = {}
    for split in ("train", "test"):
        d = sdd / split
        if not d.is_dir():
            raise FileNotFoundError(f"Expected TUTR split directory: {d}")
        clips = []
        for p in sorted(d.glob("*.txt")):
            scene, video = _clip_from_stem(p.stem)
            clips.append((scene, video, p.stem))
        if not clips:
            raise RuntimeError(f"No SDD clips found in {d}")
        out[split] = clips
    overlap = {x[2] for x in out["train"]}.intersection({x[2] for x in out["test"]})
    if overlap:
        raise RuntimeError(f"TUTR train/test clip overlap: {sorted(overlap)}")
    return out


def _flatten_scalars(obj: Any, prefix: tuple[str, ...] = ()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten_scalars(v, prefix + (str(k),))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten_scalars(v, prefix + (str(i),))
    elif isinstance(obj, (int, float, np.number)):
        yield prefix, float(obj)


def _lookup_scale(scales: Any, scene: str, video: int) -> float:
    """Robustly read OpenTraj SDD scale metadata across common YAML layouts."""
    # Common layout: scene -> videoN / index / list.
    if isinstance(scales, dict) and scene in scales:
        node = scales[scene]
        if isinstance(node, dict):
            for key in (f"video{video}", str(video), video, f"{scene}_{video}"):
                if key in node and isinstance(node[key], (int, float)):
                    return float(node[key])
        if isinstance(node, (list, tuple)) and 0 <= video < len(node):
            if isinstance(node[video], (int, float)):
                return float(node[video])

    # Flat layouts, e.g. bookstore_0: 0.0xx.
    if isinstance(scales, dict):
        for key in (f"{scene}_{video}", f"{scene}/video{video}", f"{scene}_video{video}"):
            if key in scales and isinstance(scales[key], (int, float)):
                return float(scales[key])

    # Last-resort recursive path match.
    candidates: list[tuple[tuple[str, ...], float]] = []
    scene_l = scene.lower()
    video_tokens = {str(video), f"video{video}"}
    for path, value in _flatten_scalars(scales):
        joined = "/".join(path).lower()
        if scene_l not in joined:
            continue
        tokens = {re.sub(r"[^a-z0-9]", "", p.lower()) for p in path}
        if any(re.sub(r"[^a-z0-9]", "", t) in tokens for t in video_tokens):
            candidates.append((path, value))
        elif re.search(rf"(?:^|[^0-9]){video}(?:[^0-9]|$)", joined):
            candidates.append((path, value))
    # Keep plausible meters-per-pixel values only.
    plausible = [(p, v) for p, v in candidates if 1e-4 < v < 1.0]
    if len(plausible) == 1:
        return float(plausible[0][1])
    if plausible:
        # Prefer paths containing exact videoN token.
        exact = [(p, v) for p, v in plausible if f"video{video}" in "/".join(p).lower()]
        if len(exact) == 1:
            return float(exact[0][1])
    raise KeyError(
        f"Could not resolve meters-per-pixel scale for {scene}/video{video}. "
        "Expected OpenTraj datasets/SDD/estimated_scales.yaml."
    )


def _read_annotations(path: Path, pedestrian_only: bool = True):
    ids, centers, frames = [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            try:
                track_id = int(float(parts[0]))
                xmin, ymin, xmax, ymax = map(float, parts[1:5])
                frame = int(float(parts[5]))
                lost = int(float(parts[6]))
                label = parts[9].strip('"').lower()
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: malformed annotation") from exc
            if lost != 0:
                continue
            if pedestrian_only and label != "pedestrian":
                continue
            ids.append(track_id)
            centers.append(((xmin + xmax) * 0.5, (ymin + ymax) * 0.5))
            frames.append(frame)
    if not ids:
        raise RuntimeError(f"No usable pedestrian annotations in {path}")
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(frames, dtype=np.int64),
        np.asarray(centers, dtype=np.float32),
    )


def _split_contiguous_raw(frames: np.ndarray, max_gap: int = 1):
    if len(frames) == 0:
        return []
    cuts = np.flatnonzero(np.diff(frames) > int(max_gap)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(frames)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _nearest_neighbors(ids: np.ndarray, xy_m: np.ndarray, target_index: int, topk: int):
    rel = xy_m - xy_m[target_index]
    d2 = np.sum(rel * rel, axis=1)
    d2[target_index] = np.inf
    order = np.argsort(d2, kind="stable")[:topk]
    valid = np.isfinite(d2[order])
    rr = np.zeros((topk, 2), dtype=np.float32)
    mm = np.zeros((topk,), dtype=np.float32)
    if valid.any():
        sel = order[valid]
        rr[: len(sel)] = rel[sel].astype(np.float32)
        mm[: len(sel)] = 1.0
    return rr, mm


def build_clip(
    annotation_path: Path,
    reference_path: Path,
    scene: str,
    video: int,
    clip_index: int,
    scale_m_per_px: float,
    obs_len: int,
    pred_len: int,
    frame_step: int,
    sample_stride: int,
    neighbor_topk: int,
):
    ped_ids, frames, xy_px = _read_annotations(annotation_path)
    xy_m = xy_px * float(scale_m_per_px)

    # Frame cache for social context at the exact sampled observation frames.
    frame_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fr in np.unique(frames):
        idx = np.flatnonzero(frames == fr)
        frame_map[int(fr)] = (ped_ids[idx], xy_m[idx])

    by_ped: dict[int, list[int]] = defaultdict(list)
    for idx, pid in enumerate(ped_ids.tolist()):
        by_ped[int(pid)].append(idx)

    total = int(obs_len + pred_len)
    obs, pred = [], []
    pixel_obs, pixel_pred, frame_obs, frame_pred = [], [], [], []
    neigh, neigh_mask = [], []
    seq_ids, pids, clips, scene_ids, videos = [], [], [], [], []
    local_sid = 0
    scene_index = SCENES.index(scene) if scene in SCENES else -1

    for pid, idxs in by_ped.items():
        idxs = sorted(idxs, key=lambda i: int(frames[i]))
        pf = frames[idxs]
        pm = xy_m[idxs]
        pp = xy_px[idxs]
        for s, e in _split_contiguous_raw(pf, max_gap=1):
            run_frames = pf[s:e]
            run_m = pm[s:e]
            run_px = pp[s:e]
            if len(run_frames) < 1 + (total - 1) * frame_step:
                continue

            # Downsample a continuous 30 Hz track to ~2.5 Hz using the first
            # visible frame as its sampling phase, then slide windows at 0.4 s.
            sampled = np.arange(0, len(run_frames), int(frame_step), dtype=np.int64)
            sf = run_frames[sampled]
            sm = run_m[sampled]
            sp = run_px[sampled]
            if len(sf) < total:
                continue
            for off in range(0, len(sf) - total + 1, max(1, int(sample_stride))):
                wf = sf[off: off + total]
                wm = sm[off: off + total]
                wp = sp[off: off + total]
                # Exact raw-frame spacing is required after downsampling.
                if not np.all(np.diff(wf) == int(frame_step)):
                    continue

                nr = np.zeros((obs_len, neighbor_topk, 2), dtype=np.float32)
                nm = np.zeros((obs_len, neighbor_topk), dtype=np.float32)
                for t, fr in enumerate(wf[:obs_len]):
                    ids_f, xy_f = frame_map.get(int(fr), (None, None))
                    if ids_f is None:
                        continue
                    hit = np.flatnonzero(ids_f == int(pid))
                    if len(hit) != 1:
                        continue
                    nr[t], nm[t] = _nearest_neighbors(ids_f, xy_f, int(hit[0]), neighbor_topk)

                obs.append(wm[:obs_len])
                pred.append(wm[obs_len:])
                pixel_obs.append(wp[:obs_len])
                pixel_pred.append(wp[obs_len:])
                frame_obs.append(wf[:obs_len])
                frame_pred.append(wf[obs_len:])
                neigh.append(nr)
                neigh_mask.append(nm)
                seq_ids.append(clip_index * 1_000_000 + local_sid)
                pids.append(pid)
                clips.append(clip_index)
                scene_ids.append(scene_index)
                videos.append(video)
                local_sid += 1

    if not obs:
        return None, {
            "clip": f"{scene}_{video}", "windows": 0,
            "pedestrians": len(by_ped), "scale_m_per_px": float(scale_m_per_px),
        }

    data = {
        "obs": np.stack(obs).astype(np.float32),
        "pred": np.stack(pred).astype(np.float32),
        "pixel_obs": np.stack(pixel_obs).astype(np.float32),
        "pixel_pred": np.stack(pixel_pred).astype(np.float32),
        "frame_obs": np.stack(frame_obs).astype(np.int32),
        "frame_pred": np.stack(frame_pred).astype(np.int32),
        "neighbor_rel": np.stack(neigh).astype(np.float32),
        "neighbor_mask": np.stack(neigh_mask).astype(np.float32),
        "seq_id": np.asarray(seq_ids, dtype=np.int64),
        "ped_id": np.asarray(pids, dtype=np.int64),
        "clip_index": np.asarray(clips, dtype=np.int64),
        "scene_index": np.asarray(scene_ids, dtype=np.int16),
        "video_index": np.asarray(videos, dtype=np.int16),
        "dt": np.full(len(obs), 0.4, dtype=np.float32),
        "scale_m_per_px": np.full(len(obs), float(scale_m_per_px), dtype=np.float32),
    }
    stat = {
        "clip": f"{scene}_{video}",
        "windows": int(len(obs)),
        "pedestrians": int(len(by_ped)),
        "scale_m_per_px": float(scale_m_per_px),
        "neighbor_valid_ratio": float(data["neighbor_mask"].mean()),
        "reference_image": str(reference_path.resolve()),
        "annotation_file": str(annotation_path.resolve()),
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
    }
    return data, stat


def combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise RuntimeError("No raw SDD windows were extracted")
    keys = tuple(parts[0].keys())
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def main():
    ap = argparse.ArgumentParser(description="Prepare raw SDD trajectories with exact pixel anchors for TMM visual-context experiments.")
    ap.add_argument("--annotations_root", required=True)
    ap.add_argument("--tutr_root", required=True)
    ap.add_argument("--scales_yaml", required=True)
    ap.add_argument("--out_dir", default="data/sdd_raw_visual")
    ap.add_argument("--obs_len", type=int, default=8)
    ap.add_argument("--pred_len", type=int, default=12)
    ap.add_argument("--frame_step", type=int, default=12)
    ap.add_argument("--sample_stride", type=int, default=1)
    ap.add_argument("--neighbor_topk", type=int, default=8)
    args = ap.parse_args()

    annotations_root = Path(args.annotations_root).resolve()
    tutr_root = Path(args.tutr_root).resolve()
    scales_path = Path(args.scales_yaml).resolve()
    if not scales_path.exists():
        raise FileNotFoundError(scales_path)
    scales = yaml.safe_load(scales_path.read_text(encoding="utf-8"))
    split_clips = _load_split_clips(tutr_root)

    # Stable clip index independent of split iteration order.
    all_names = sorted({name for rows in split_clips.values() for _, _, name in rows})
    clip_index_map = {name: i + 1 for i, name in enumerate(all_names)}

    out_fold = Path(args.out_dir) / "sdd"
    out_fold.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": "raw Stanford Drone Dataset annotations + TUTR recording split + OpenTraj estimated scales",
        "protocol": "TUTR/SocialVAE recording train-test split; raw annotation tracks downsampled every 12 frames; sliding 8/12 windows; held-out test locked during model selection",
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "frame_step": int(args.frame_step),
        "sample_stride": int(args.sample_stride),
        "dt_seconds": 0.4,
        "neighbor_topk": int(args.neighbor_topk),
        "scales_yaml": str(scales_path),
        "clip_map": {},
        "splits": {},
    }

    for split in ("train", "test"):
        parts, stats = [], []
        for scene, video, name in split_clips[split]:
            clip_index = clip_index_map[name]
            ann = annotations_root / scene / f"video{video}" / "annotations.txt"
            ref = annotations_root / scene / f"video{video}" / "reference.jpg"
            if not ann.exists() or not ref.exists():
                raise FileNotFoundError(f"Missing raw SDD assets for {name}: {ann} / {ref}")
            scale = _lookup_scale(scales, scene, video)
            if not (1e-4 < scale < 1.0):
                raise ValueError(f"Implausible SDD scale for {name}: {scale}")
            data, stat = build_clip(
                ann, ref, scene, video, clip_index, scale,
                args.obs_len, args.pred_len, args.frame_step,
                args.sample_stride, args.neighbor_topk,
            )
            stat["clip_index"] = int(clip_index)
            stats.append(stat)
            manifest["clip_map"][str(clip_index)] = {
                "name": name, "scene": scene, "video": int(video),
                "reference_image": str(ref.resolve()),
                "annotation_file": str(ann.resolve()),
                "scale_m_per_px": float(scale),
                "split": split,
            }
            if data is not None:
                parts.append(data)
        merged = combine(parts)
        np.savez_compressed(out_fold / f"{split}.npz", **merged)
        manifest["splits"][split] = {
            "num_clips": int(len(stats)),
            "num_windows": int(len(merged["obs"])),
            "neighbor_valid_ratio": float(merged["neighbor_mask"].mean()),
            "stats": stats,
        }
        print(
            f"{split}: clips={len(stats)} windows={len(merged['obs'])} "
            f"obs={merged['obs'].shape} pixel={merged['pixel_obs'].shape} "
            f"neighbor_valid={merged['neighbor_mask'].mean():.4f}"
        )

    train_clips = {c[2] for c in split_clips["train"]}
    test_clips = {c[2] for c in split_clips["test"]}
    if train_clips & test_clips:
        raise RuntimeError("Raw SDD train/test clip leakage detected")
    (out_fold / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved raw visual SDD dataset to {out_fold.resolve()}")


if __name__ == "__main__":
    main()
