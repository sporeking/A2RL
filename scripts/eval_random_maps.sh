# BASE_MODEL_NAME="ramdom-ABL-PPO-large-real"
BASE_MODEL_NAME="20250110-discover-DQN-small-easy-random"
# BASE_MODEL_NAME="20250119-discover-DQN-small-large-random"
ENV="MiniGrid-ConfigWorld-Random"
CONFIGMAP="test_random_small_maps.config"
FIRST=1
LAST=5

for i in $(seq $FIRST $LAST); do
    MODEL_NAME="$BASE_MODEL_NAME-${i}"
    MODEL_CONFIG_FOLDER="config/$MODEL_NAME"
    python evaluate_random_maps.py --model $MODEL_NAME --env $ENV --seed $i --configmap $CONFIGMAP
done