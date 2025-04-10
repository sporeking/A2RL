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

## Some details

The model and training data is in `scripts/storage` folder.  
The config files (containing the structure of initial state machine and discovered state machine) are in `scripts/config` folder. You can find the symbols (mutations) in the folder. 

## Referenced Projects

This project references the following open-source projects:

- [TORCH-AC](https://github.com/lcswillems/torch-ac) - Recurrent and multi-process PyTorch implementation of deep reinforcement Actor-Critic algorithms A2C and PPO.
