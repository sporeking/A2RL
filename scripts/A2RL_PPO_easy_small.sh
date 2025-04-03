#!/bin/bash

NUMS=10
DEVICE_ID=0 
DELETE_OLD_MODELS=0 
BASE_MODEL_NAME="test_A2RL_easy_small_111" 
CONFIGMAP="easy_small_maps.config" 
ENV="MiniGrid-ConfigWorld-v0" 
# PYTHON=/home/sporeking/miniconda3/envs/py312/bin/python
CURRICULUM_1_STEPS=100000
CURRICULUM_2_STEPS=200000
CURRICULUM_3_STEPS=300000
DISCOVER_STEPS=100000 
###################################

# init config.
START_CONFIG_CONTENT="graph:  
  nodes:
    - id: 0
      next: 0
    - id: 1
      next: 1
    - id: 2
      next: 1
  # 0 == die, 1 == reward. 
  edges:
    - from: 2
      to: 1
      id: 0
      with_agent: 1
    - from: 2
      to: 0
      id: 1
      with_agent: 0
  start_node: 2
agent_num: 1"

LR=0.00006
DISCOUNT=0.99
ALGO=ppo
EPOCHS=8
BATCH_SIZE=128
FRAMES_PER_PROC=512

for i in $(seq 1 $NUMS); do
  MODEL_NAME="$BASE_MODEL_NAME-${i}"
  MODEL_CONFIG_FOLDER="config/$MODEL_NAME"

  if [ "$DELETE_OLD_MODELS" == "1" ]; then
    echo "Warning: Deleting old model and config..."
    rm -rf $MODEL_CONFIG_FOLDER
    rm -rf storage/$MODEL_NAME
  else
    echo "Use old model and config"
  fi
  
  if [ ! -d $MODEL_CONFIG_FOLDER ]; then
    echo "The folder $MODEL_CONFIG_FOLDER does not exist, creating it..."
    mkdir -p $MODEL_CONFIG_FOLDER/task1
    touch $MODEL_CONFIG_FOLDER/task1/config.yaml
    NEW_TASK_CONFIG=$MODEL_CONFIG_FOLDER/task1/config.yaml
    printf "%s\n" "$START_CONFIG_CONTENT" >> $NEW_TASK_CONFIG
  else
    echo "The folder $MODEL_CONFIG_FOLDER already exists."
  fi

  CUDA_VISIBLE_DEVICES=$DEVICE_ID python discover.py --task-config task1 --discover 0 --algo $ALGO --env $ENV --lr $LR  --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames $CURRICULUM_1_STEPS --seed $i --configmap $CONFIGMAP --curriculum 1 --discover-steps $DISCOVER_STEPS
  if [ $? -gt 4 ]; then
    echo "Error during task 1, stopping the script."
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=$DEVICE_ID python discover.py --task-config task1 --discover 1 --algo $ALGO --env $ENV --lr $LR  --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames $CURRICULUM_2_STEPS --seed $i --configmap $CONFIGMAP --curriculum 2 --discover-steps $DISCOVER_STEPS
  if [ $? -gt 4 ]; then
    echo "Error during task 2, stopping the script."
    exit 1
  fi

  CUDA_VISIBLE_DEVICES=$DEVICE_ID python discover.py --task-config task2 --discover 1 --algo $ALGO --env $ENV --lr $LR --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames $CURRICULUM_3_STEPS --seed $i --configmap $CONFIGMAP --curriculum 3 --discover-steps $DISCOVER_STEPS
  if [ $? -gt 4 ]; then
    echo "Error during task 3, stopping the script."
    exit 1
  fi
done
