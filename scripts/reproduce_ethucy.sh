#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/tmm_stage3e_ethucy_final logs
FOLDS="${FOLDS:-eth hotel univ zara1 zara2}"
SEEDS="${SEEDS:-42 43 44}"
VARIANTS="${VARIANTS:-multihyp_social}"
DATA="${ETHUCY_DATA_ROOT:-data/ethucy_5fold_social}"
for variant in $VARIANTS; do
 for fold in $FOLDS; do
  if [[ ! -f "$DATA/$fold/train.npz" || ! -f "$DATA/$fold/test.npz" ]]; then
   echo "Missing ETH/UCY train/test for $fold. Prepare authorized data first." >&2; exit 2
  fi
  for seed in $SEEDS; do
   path="runs/tmm_stage3e_ethucy_final/$(basename "$DATA")/$variant/$fold/seed_${seed}/summary.json"
   if [[ -f "$path" && "${FORCE_RERUN:-0}" != 1 ]]; then
    echo "EXISTS $path — not overwriting (set FORCE_RERUN=1 to rerun)"
    continue
   fi
   tag="${variant}_${fold}_${seed}"
   echo "[TRAIN] $tag"
   python train.py --variant "$variant" --data_root "$DATA" \
    --test_set "$fold" --val_source train_group --val_ratio 0.1 \
    --epochs 100 --batch_size 512 --lr 0.01 --seed "$seed" \
    --ranking_loss hard_ce --checkpoint_metric auto \
    --cls_lambda 0.2 --nll_lambda 0.1 --num_modes 20 --best_k 20 \
    --use_augment --eval_views 1 --out_dir runs/tmm_stage3e_ethucy_final \
    2>&1 | tee "logs/${tag}.log"
  done
 done
 done
