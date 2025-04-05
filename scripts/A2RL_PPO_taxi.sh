#!/bin/bash

NUMS=10
DEVICE_ID=0
DELETE_OLD_MODELS=0         # whether to delete old models and configs.
BASE_MODEL_NAME="taxi-test" # Your model name.
ENV="Taxi-v0"
DISCOVER_STEPS=80000
TOTAL_STEPS=200000
CONTRAST_FUNC="HIST"

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
    printf "%s\n" "$START_CONFIG_CONTENT" >>$NEW_TASK_CONFIG
  else
    echo "The folder $MODEL_CONFIG_FOLDER already exists."
  fi

  python discover.py --task-config task1 --discover 1 --algo $ALGO --env $ENV --lr $LR --model $MODEL_NAME --discount $DISCOUNT --epochs $EPOCHS --frames-per-proc $FRAMES_PER_PROC --frames $TOTAL_STEPS --seed $i --curriculum 1 --discover-steps $DISCOVER_STEPS --contrast $CONTRAST_FUNC
done
