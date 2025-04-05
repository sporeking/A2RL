import argparse
import yaml
import time
import copy
import datetime
import torch_ac
import tensorboardX
from torchvision import transforms
import sys
import networkx as nx
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import gymnasium as gym
from typing import Optional, Callable
from skimage.metrics import structural_similarity as ssim

import utils
from utils import *
from utils import device
from utils.KB import RGB2GARY_ROI
from model import ACModel, CNN, QNet

def obs_To_mutation(pre_obs, obs, preprocess_obss):
    pre_image_data=preprocess_obss([pre_obs], device=device).image
    image_data=preprocess_obss([obs], device=device).image
    input_tensor = image_data - pre_image_data
    input_tensor = numpy.squeeze(input_tensor)
    # input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    return input_tensor

def test_once(
    G: nx.Graph,
    start_env: gym.Env, 
    mutation_buffer: list, 
    start_node: int,
    max_steps: int,
    env_key: str,
    anomaly_detector: Optional[Callable], 
    preprocess_obss: Optional[Callable],
    args,
    ):
    env = copy_env(start_env, env_key, curriculum=args.curriculum)
    env.reset()

    current_state = start_node
    obss = env.gen_obs()
    pre_obss = obss
    stop_env = copy_env(env, env_key)
    stop_obss = obss

    episode_return = 0
    episode_num_frames = 0

    # print(len(mutation_buffer))
    # for i in range(len(mutation_buffer)):

    #     plt.imshow(mutation_buffer[i][1])
    #     plt.show()
    
    for i in range(max_steps):
        mutation = obs_To_mutation(pre_obss, obss, preprocess_obss)
        mutation = mutation.cpu().numpy().astype(numpy.uint8)
        _, mutation_roi_list = RGB2GARY_ROI(mutation)
        # anomaly_mutation = transforms.ToTensor()(mutation).cuda().unsqueeze(0)
        # if anomaly_detector(anomaly_mutation)[0, 0] < anomaly_detector(anomaly_mutation)[0, 1]:
        for mutation_roi in mutation_roi_list:
            # if not anomaly_detector.is_known_roi(mutation_roi, add_to_buffer=False):
            #     continue
            is_anomaly = False
            for node_num, node_mutation in mutation_buffer:
                if anomaly_detector.contrast(node_mutation, mutation_roi, return_bool=True):
                    # print(f"turn to state {node_num}")
                    current_state = node_num
                    # print(current_state)
                    # plt.imshow(mutation)
                    # plt.show()
                    # plt.imshow(mutation_roi, cmap='gray')
                    # plt.show()
                    stop_env = copy_env(env, env_key)
                    stop_obss = copy.deepcopy(obss)
                    is_anomaly = True
                    break
                if is_anomaly:
                    break

        preprocessed_obss = preprocess_obss([obss], device=device)
        if args.algo == 'dqn':
            actions = G.nodes[current_state]['state'].agent.select_action(preprocessed_obss,0)
        else:
            with torch.no_grad():
                dist, _ = G.nodes[current_state]['state'].agent.acmodel(preprocessed_obss)
            actions = dist.sample()
        pre_obss = obss
        obss, rewards, terminateds, truncateds, infos = env.step(actions if isinstance(actions, int) else actions.item())
        dones = terminateds | truncateds
        
        episode_return += rewards
        episode_num_frames += 1
        
        if dones:
            if rewards > 0:
                current_state = 1
            # elif rewards < 0:
                # current_state = 0
            return episode_return, episode_num_frames, current_state, stop_env, stop_obss
    return episode_return, episode_num_frames, current_state, stop_env, stop_obss

def test(
    G: nx.Graph,
    start_env: gym.Env,
    start_node: int,
    episodes: int,
    max_steps_per_episode: int,
    env_key: str,
    preprocess_obss: Optional[Callable],
    anomaly_detector: Optional[Callable],
    args,
):
    test_logs = {"num_frames_per_episode": [], "return_per_episode": []}
    env = copy_env(start_env, env_key)
    mutation_buffer = []
    for node, stateNode in G.nodes(data=True):
        # print(stateNode['state'].agent)
        if stateNode['state'].mutation is not None:
            mutation_buffer.append((node, stateNode['state'].mutation))
    
    log_done_counter = 0
    log_episode_return = torch.zeros(1, device=device)
    log_episode_num_frames = torch.zeros(1, device=device) 

    stop_env_list = []
    stop_obs_list = []
    stop_state_list = []

    for episode in range(episodes):
        episode_return, episode_num_frames, current_state, stop_env, stop_obss = test_once(
            G, start_env, mutation_buffer, start_node, max_steps_per_episode, env_key, anomaly_detector, preprocess_obss, args
        )
        print("Episode: ", episode, "Return: ", episode_return, "Num_frames: ", episode_num_frames)
        test_logs["num_frames_per_episode"].append(episode_num_frames)
        test_logs["return_per_episode"].append(episode_return)
        log_episode_return += episode_return
        log_episode_num_frames += episode_num_frames
        log_done_counter += 1

        stop_env_list.append(stop_env)
        stop_obs_list.append(stop_obss.copy())
        stop_state_list.append(current_state)

    return_per_episode = utils.synthesize(test_logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(test_logs["num_frames_per_episode"])
    print(test_logs)

    counter = collections.Counter(stop_state_list)
    stop_state, _ = counter.most_common(1)[0]
    stop_state_index = stop_state_list.index(stop_state)
    stop_env = stop_env_list[stop_state_index]
    stop_obss = stop_obs_list[stop_state_index]
    return return_per_episode, num_frames_per_episode, stop_state, stop_env, stop_obss

def ddm_decision(
    G: nx.Graph,
    start_env: gym.Env,
    env_key: str,
    max_decision_steps: int,
    node_probability_list: list, 
    preprocess_obss: Optional[Callable],
    anomaly_detector: Optional[Callable],
    drift_rate: float,
    boundary_separation: float,
    starting_point: float,
    args,
    # non_decision_time: float,
):
    decision_steps = 0
    position = starting_point
    mutation_buffer = []
    for node, stateNode in G.nodes(data=True):
        if stateNode['state'].mutation is not None:
            mutation_buffer.append((node, stateNode['state'].mutation))

    stop_env_list = []
    stop_obs_list = []
    stop_state_list = []
    total_return = 0

    return_mean_list = [[]] * len(G.nodes)
    
    while decision_steps < max_decision_steps:
        test_start_node = numpy.random.choice(len(node_probability_list), 1,  p=node_probability_list)[0]

        episode_return, episode_num_frames, stop_state, stop_env, stop_obss = test_once(
            G=G, 
            start_env=start_env, 
            mutation_buffer=mutation_buffer, 
            start_node=test_start_node, 
            max_steps=256,
            env_key=env_key, 
            preprocess_obss=preprocess_obss, 
            anomaly_detector=anomaly_detector,
            args=args
        )
        print("Decision step: ", decision_steps, "Return: ", episode_return, "Num_frames: ", episode_num_frames)
        print("Start state", test_start_node, "Stop state: ", stop_state)

        if len(return_mean_list[test_start_node]) == 0:
            return_mean_list[test_start_node].append(episode_return)
        else:
            n = len(return_mean_list[test_start_node])
            mean_return = sum(return_mean_list[test_start_node]) / float(n)
            if n >= 3 and mean_return >= 0.7:
                return False, decision_steps, None, None, None, return_per_episode
        
        total_return += episode_return
        stop_env_list.append(stop_env)
        stop_obs_list.append(stop_obss)
        stop_state_list.append(stop_state)

        decision_steps += 1
        
        drift = drift_rate * episode_return + numpy.random.normal(-0.03, 0.03)
        position += drift
        print("position: ", position)
        return_per_episode = total_return / decision_steps
        if position >= boundary_separation:
            print("Successful test, no need to discover.")
            return False, decision_steps, None, None, None, return_per_episode
        elif position <= -boundary_separation:
            print("Failed test, need to discover.")
            counter = collections.Counter(stop_state_list)
            # stop_state, _ = counter.most_common(1)[0]
            stop_state = max(stop_state_list)
            stop_state_index = stop_state_list.index(stop_state)
            stop_env = stop_env_list[stop_state_index]
            stop_obss = stop_obs_list[stop_state_index]
            print("Stop state: ", stop_state)
            return True, decision_steps, stop_state, stop_env, stop_obss, return_per_episode

    return_per_episode = total_return / decision_steps
    if position > 0:
        print("Arrive at boundary and position > 0, no need to discover.")
        return False, decision_steps, None, None, None, return_per_episode
    else:
        print("Arrive at boundary and position <= 0, need to discover.")
        counter = collections.Counter(stop_state_list)
        stop_state, _ = counter.most_common(1)[0]
        stop_state_index = stop_state_list.index(stop_state)
        stop_env = stop_env_list[stop_state_index]
        stop_obss = stop_obs_list[stop_state_index]
        return True, decision_steps, stop_state, stop_env, stop_obss, return_per_episode