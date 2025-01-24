# A2RL (Abductive Abstract Reinforcement Learning) 

## Environment Installation

### MiniGrid installation

```
cd Minigrid-master
pip install -e .
```

### Torch-ac Installation

```
cd torch-ac
pip3 install -e .
```

### other packages

```
pip install -r requirements.txt
```

## Run

Example: A2RL with PPO on Minigrid-easy-small:

```
cd scripts
bash ./A2RL_PPO_easy_small.sh
```

A2RL with PPO on Taxi:
```
cd scripts
bash ./A2RL_PPO_taxi.sh
```
