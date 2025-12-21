#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
TASK=ShadowHandRandomLoadVision
ALGO=ppo
SEED=0
BACKBONE=pn 
BASE_MODEL_DIR="1117/vase_1654_010_seed0"
HEAD="--headless --vision --test"

CKPTS=(3000 4000 5000 6000 7000)
NFINGERS=(0 1 2)

gpu_idx=0
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
NUM_RUNS=5
for run in $(seq 1 $NUM_RUNS); do 
  for nf in "${NFINGERS[@]}"; do
    for ck in "${CKPTS[@]}"; do
      RL_DEV="cuda:${gpu_idx}"
      SIM_DEV="cuda:${gpu_idx}"
      SEED=$RANDOM

      outdir="logs/${TASK}/nf${nf}_ck${ck}"
      mkdir -p "$outdir"

      echo ">>> running: num_finger_contact=${nf}, ckpt=${ck}, gpu=${gpu_idx} -> ${outdir}"

      python train.py \
        --task="${TASK}" \
        --algo="${ALGO}" \
        --seed="${SEED}" \
        --rl_device="${RL_DEV}" \
        --sim_device="${SIM_DEV}" \
        --num_finger_contact="${nf}" \
        --model_dir="${BASE_MODEL_DIR}/model_${ck}.pt" \
        ${HEAD} \
        --backbone_type "${BACKBONE}" \
        2>&1 | tee "${outdir}/train.log"
    done
  done
done
