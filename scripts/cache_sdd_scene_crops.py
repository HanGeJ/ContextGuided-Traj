from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def crop_with_padding(img: Image.Image, center_xy: tuple[float, float], side_px: int) -> Image.Image:
    side_px = max(8, int(side_px))
    half = side_px / 2.0
    cx, cy = center_xy
    left, top = int(round(cx - half)), int(round(cy - half))
    right, bottom = left + side_px, top + side_px
    # PIL pads out-of-image regions with black, which is deterministic and
    # keeps the geometric center fixed near image boundaries.
    return img.crop((left, top, right, bottom))


def main():
    ap = argparse.ArgumentParser(description="Cache target-centered static SDD reference-image crops as uint8 memmaps.")
    ap.add_argument("--data_root", default="data/sdd_raw_visual")
    ap.add_argument("--fold", default="sdd")
    ap.add_argument("--crop_meters", type=float, default=16.0,
                    help="Physical side length of local scene field-of-view.")
    ap.add_argument("--output_size", type=int, default=160)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root) / args.fold
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    clip_map = manifest["clip_map"]
    tag = f"m{args.crop_meters:g}_{args.output_size}px"
    out_dir = root / f"scene_crops_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "crop_meters": float(args.crop_meters),
        "output_size": int(args.output_size),
        "splits": {},
    }

    for split in ("train", "test"):
        npz = root / f"{split}.npz"
        with np.load(npz, allow_pickle=False) as d:
            pixel_obs = np.asarray(d["pixel_obs"], dtype=np.float32)
            clip_index = np.asarray(d["clip_index"], dtype=np.int64)
            scales = np.asarray(d["scale_m_per_px"], dtype=np.float32)
        n = len(pixel_obs)
        out_path = out_dir / f"{split}.npy"
        if out_path.exists() and not args.overwrite:
            existing = np.load(out_path, mmap_mode="r")
            if existing.shape == (n, 3, args.output_size, args.output_size):
                print(f"SKIP existing {out_path} {existing.shape}")
                report["splits"][split] = {"samples": n, "path": str(out_path.resolve()), "skipped": True}
                continue

        mm = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.uint8,
            shape=(n, 3, args.output_size, args.output_size),
        )
        image_cache: dict[int, Image.Image] = {}
        side_px_values = []
        for i in range(n):
            ci = int(clip_index[i])
            if ci not in image_cache:
                ref_path = Path(clip_map[str(ci)]["reference_image"])
                image_cache[ci] = Image.open(ref_path).convert("RGB")
            img = image_cache[ci]
            scale = float(scales[i])
            side_px = max(16, int(round(float(args.crop_meters) / max(scale, 1e-8))))
            side_px_values.append(side_px)
            anchor = tuple(map(float, pixel_obs[i, -1]))
            crop = crop_with_padding(img, anchor, side_px)
            crop = crop.resize((args.output_size, args.output_size), Image.Resampling.BILINEAR)
            arr = np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1)
            mm[i] = arr
            if (i + 1) % 5000 == 0 or i + 1 == n:
                print(f"{split}: cached {i + 1}/{n}", flush=True)
        mm.flush()
        report["splits"][split] = {
            "samples": int(n),
            "path": str(out_path.resolve()),
            "side_px_mean": float(np.mean(side_px_values)),
            "side_px_min": int(np.min(side_px_values)),
            "side_px_max": int(np.max(side_px_values)),
            "skipped": False,
        }
        for img in image_cache.values():
            img.close()

    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SCENE_CROP_CACHE_OK tag={tag}")


if __name__ == "__main__":
    main()
