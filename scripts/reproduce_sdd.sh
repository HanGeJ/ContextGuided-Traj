#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmd="${1:---check}"
GROOT="${SDD_GENERATOR_ROOT:-runs/tmm_stage3b_raw_control/sdd_raw_visual}"
VROOT="${SDD_VISUAL_ROOT:-runs/tmm_stage3d_decoupled}"
RAW="${SDD_DATA_ROOT:-data/sdd_raw_visual}"
failed=0
for s in 42 43 44; do
 g="$GROOT/multihyp_social/sdd/seed_${s}/best_model.pt"
 v="$VROOT/decoupled_social_visual_adapter/m16_160px/mobilenet_v3_small/frozen/pretrained/cscale_0.2/soft_t0.05_d0.0001/seed_${s}/geometry_best.pt"
 for file in "$g" "$v"; do
  if [[ ! -f "$file" ]]; then echo "MISSING $file"; failed=1; else echo "OK $file"; fi
 done
 done
for file in "$RAW/sdd/train.npz" "$RAW/sdd/test.npz" "$RAW/sdd/scene_crops_m16_160px/train.npy" "$RAW/sdd/scene_crops_m16_160px/test.npy"; do
 if [[ ! -f "$file" ]]; then echo "MISSING $file"; failed=1; else echo "OK $file"; fi
done
if [[ "$failed" != 0 ]]; then echo "SDD assets incomplete; refusing evaluation" >&2;exit 3;fi
if [[ "$cmd" == --check ]]; then echo "SDD frozen assets present. No test evaluation performed.";exit 0; fi
if [[ "$cmd" != --evaluate ]];then echo "Usage: bash scripts/reproduce_sdd.sh --check|--evaluate" >&2;exit 2;fi
if [[ "${ACKNOWLEDGE_SDD_HELDOUT:-0}" != 1 ]];then
 echo "Set ACKNOWLEDGE_SDD_HELDOUT=1 only for the pre-registered final test. No test-based tuning." >&2;exit 4
fi
python evaluate_stage3e_locked_sdd.py --data_root "$RAW" --raw_control_root "$GROOT" --stage3d_root "$VROOT" --out runs/tmm_stage3e_locked_sdd_test.json --lock_file runs/tmm_stage3e_sdd_test_lock.json
