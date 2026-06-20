#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate second_aiffel
fi

mkdir -p run_logs

DATA_DIR="data_no_iemocap"
if [[ ! -f "${DATA_DIR}/full_dataset_fold1.csv" || ! -f "${DATA_DIR}/full_dataset_fold2.csv" ]]; then
  echo "Missing ${DATA_DIR}/full_dataset_fold1.csv or fold2.csv"
  echo "Run: python setup_no_iemocap_data.py --data-dir data --output-dir data_no_iemocap --seed 42"
  exit 1
fi

ET_CKPT="./checkpoints/et_predictor2_seed123"
if ! ls "${ET_CKPT}"* >/dev/null 2>&1; then
  echo "Missing ET checkpoint: ${ET_CKPT}*"
  echo "Run: python setup_et_models.py --et2-checkpoint ./checkpoints/et_predictor2_seed123"
  exit 1
fi

COMMON_ARGS=(
  mse
  --data-dir "${DATA_DIR}"
  --maxlen 200
  --batch-size 16
  --train-epochs 10
  --optim adamw_torch
  --save-strategy no
  --no-save-final-model
  --no-load-best-model-at-end
  --seed 42
)

echo "===== 1/4 distilbert baseline ====="
python train_model.py distilbert "${COMMON_ARGS[@]}" \
  2>&1 | tee run_logs/distilbert_baseline_seed42.log

echo "===== 2/4 distilbert gazeconcat TRT only ====="
python train_model.py distilbert "${COMMON_ARGS[@]}" \
  --gaze-fusion concat \
  --et2-checkpoint "${ET_CKPT}" \
  --features-used 0,0,0,1,0 \
  --fp-dropout 0.1,0.3 \
  2>&1 | tee run_logs/distilbert_gazeconcat_trt_seed42.log

echo "===== 3/4 xlmroberta-base baseline ====="
python train_model.py xlmroberta-base "${COMMON_ARGS[@]}" \
  2>&1 | tee run_logs/xlmroberta_base_baseline_seed42.log

echo "===== 4/4 xlmroberta-base gazeconcat TRT only ====="
python train_model.py xlmroberta-base "${COMMON_ARGS[@]}" \
  --gaze-fusion concat \
  --et2-checkpoint "${ET_CKPT}" \
  --features-used 0,0,0,1,0 \
  --fp-dropout 0.1,0.3 \
  2>&1 | tee run_logs/xlmroberta_base_gazeconcat_trt_seed42.log

echo "ALL_DONE"
