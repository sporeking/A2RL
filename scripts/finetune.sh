#!/bin/bash

# PYTHON=/home/sporeking/miniconda3/envs/py312/bin/python
NUMS=50 
DEVICE_ID=0 
DELETE_OLD_MODELS=0 
MODEL_NAME="ramdom-ABL-PPO-large-real-1" 
CONFIGMAP="test_random_big_maps.config"
ENV="MiniGrid-ConfigWorld-v0" 
TOTAL_STEPS=200000
CONTRAST_FUNC="SSIM"

LR=0.00006
DISCOUNT=0.995
ALGO=ppo
EPOCHS=8
BATCH_SIZE=128
FRAMES_PER_PROC=512

for FIXED_MAP in $(seq 1 $NUMS); do
  CUDA_VISIBLE_DEVICES=$DEVICE_ID python finetune.py --fixed-map $FIXED_MAP --task-config task3 --algo $ALGO --env $ENV --lr $LR --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames $TOTAL_STEPS --seed $FIXED_MAP --configmap $CONFIGMAP --curriculum 3 --contrast $CONTRAST_FUNC
  if [ $? -gt 4 ]; then
    echo "Error during task 3, stopping the script."
    exit 1
  fi
done