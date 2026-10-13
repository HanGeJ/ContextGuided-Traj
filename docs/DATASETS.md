# External data setup and split protocol

No ETH–UCY or SDD raw frames, annotation files, TUTR caches, processed trajectory datasets or third-party weights are redistributed in this repository.

## ETH/UCY

Acquire the ETH/UCY datasets from authorized sources and place their `true_pos_.csv` files at:

```text
data/eth_ucy_raw/
  eth/univ/true_pos_.csv
  eth/hotel/true_pos_.csv
  ucy/univ/students001/true_pos_.csv
  ucy/univ/students003/true_pos_.csv
  ucy/univ/uni_examples/true_pos_.csv
  ucy/zara/zara01/true_pos_.csv
  ucy/zara/zara02/true_pos_.csv
```

The archived preprocessor explicitly excludes `zara03` and uses sliding windows, 8 observation and 12 future samples, four observation-time nearest neighbors per frame. It generates `train.npz` and `test.npz` for five leave-one-scene-out folds. Validation is constructed only from training-group raw recordings using `seq_id` groups (not held-out test). The actual count can differ from 10% because whole groups are held out; this is documented, not silently changed.

## SDD

- Raw Stanford Drone Dataset (reference images + annotations) acquired externally.
- TUTR/SocialVAE-aligned recording split from its relevant official source.
- Recording-specific meters-per-pixel YAML from the appropriate scale source (the historical project used OpenTraj scales).
- The conversion script `prepare_sdd_raw_visual.py` requires these three inputs:

```bash
python prepare_sdd_raw_visual.py \
  --annotations_root /path/to/Stanford_Drone_Dataset \
  --tutr_root /path/to/authorized/TUTR/split \
  --scales_yaml /path/to/scale_file.yaml \
  --out_dir data/sdd_raw_visual
```

Check your local TUTR split directory with `_load_split_clips` in the original script; its exact file names are defined there. The crop cache must be generated with the frozen `m16_160px` physical field-of-view specification. The original crop-cache implementation is included under `scripts/cache_sdd_scene_crops.py` but accepts file layout parameters; inspect `--help` before running.

SDD paper metrics are **pixels** for Table II and **meters** for visual/context ablations. Pretrained MobileNetV3-Small weights are provided through torchvision's official download at runtime and subject to its terms.

The locked SDD configuration is selected on `train_group` validation before held-out testing, then its evaluation results are sealed. Do not use held-out SDD tests to select the model.
