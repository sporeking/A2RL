import argparse
from typing import Optional
import yaml
import time
import torch_ac
import tensorboardX
from torchvision import transforms
import sys
import networkx as nx
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

import utils
from utils import *
from utils import device
from utils.process import contrast_ssim, contrast_hist
from model import ACModel, CNN, QNet

from graph_test import test, ddm_decision, test_once
from utils.anomaly import BoundaryDetector, BoundaryDetectorSSIM, ClusterAnomalyDetector
import math
from utils.KB import RGB2GARY_ROI, obs_To_state

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="MiniGrid-ConfigWorld-v0",
                    help="name of the environment (REQUIRED)")
parser.add_argument("--task-config", default="task3",
                    help="task config file (REQUIRED)")
parser.add_argument("--model", required=True, 
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--configmap", default="test_random_small_maps.config", type=str,
                    help="the name of the map config file.")
parser.add_argument("--discount", type=float, default=0.995,
                    help="discount factor (default: 0.99)")
### below are useless
parser.add_argument("--discover", required=False, type=int, default=0,
                    help="if this task need to discover new state")
parser.add_argument("--algo", required=False, type=str, default="ppo",
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--frames-per-proc", type=int, default=128, 
                    help="number of frames per process before update")
parser.add_argument("--epochs", type=int, default=8,)
parser.add_argument("--frames", type=int, default=int(1e6))
parser.add_argument("--seed", type=int, default=1,
                    help="random seed")
parser.add_argument("--curriculum", type=int, default=3,)
parser.add_argument("--episodes-per-env", type=int, default=3,)
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--buffer-size", type=int, default=10000,
                    help="buffer size for dqn")
parser.add_argument("--target-update", type=int, default=5,
                    help="frequency to update target net")
parser.add_argument("--batch-size", type=int, default=128,
                    help="batch size for PPO (default: 256)")


class stateNode():
    def __init__(self,
                 id,
                 mutation = None,
                 agent: ACModel = None,
                 env_image = None):
        self.id = id
        self.mutation = mutation
        self.agent = agent
        self.env_image = env_image

def main():
    args = parser.parse_args()
    normal_buffer_path = "config/" + args.model + "/buffer/"
    task_path=  os.path.join("config", args.model, args.task_config)
    task_config_path = os.path.join("config", args.model, args.task_config, "config.yaml")
    task_number = 3
    with open(task_config_path, "r") as stream:
        task_config = yaml.safe_load(stream)

    G = nx.DiGraph()

    for node_id in task_config['graph']['nodes']:
        # print(stateNode(node_id))
        G.add_node(node_id, state=stateNode(node_id, agent=None, mutation=None))
    for edge in task_config['graph']['edges']:
        G.add_edge(edge["from"], edge["to"])

    start_node = task_config['graph']['start_node']
    model_name = args.model
    model_dir = utils.get_model_dir(model_name)
    txt_logger = utils.get_txt_logger(model_dir)
    csv_path = os.path.join(model_dir, "random_results.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    csv_writer = csv.writer(csv_file)

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))
    utils.seed(args.seed)
    print("Seed:", args.seed)

    envs = [utils.make_env(args.env, args.seed + 10000, curriculum=3, config_path=args.configmap, fixed_map=0)]
    initial_obs, _ = envs[0].reset()

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    # txt_logger.info("status: {}\n".format(status['model_state']))
    txt_logger.info("Training status loaded\n")

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    agent_num = task_config['agent_num']
    acmodels = []
    for i in range(agent_num):
        if args.algo == "a2c" or args.algo == "ppo":
            acmodel = ACModel(obs_space, envs[0].action_space, args.text)
        elif args.algo == "dqn":
            acmodel = QNet(obs_space, envs[0].action_space, args.text)
        if "model_state" in status and status["model_state"][i] is not None:
            print(f"Agent {i} loaded old model.")
            acmodel.load_state_dict(status["model_state"][i])
        else:
            print(f"This model {i} is None. No load.")
        acmodel.to(device)
        acmodels.append(acmodel)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodels[0]))

    algos=[]
    algos.append(None)
    algos.append(None)
    for i in range(agent_num):
        # Load algo
        if args.algo == "a2c":
            algo = torch_ac.A2CAlgo(envs, acmodels[i], device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_alpha, args.optim_eps, preprocess_obss)
        elif args.algo == "ppo":
            algo = torch_ac.PPOAlgo(envs, acmodels[i], device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
        elif args.algo == "dqn":
            algo = torch_ac.DQNAlgo(envs, acmodels[i], device, args.frames_per_proc, args.discount, args.lr,
                                    args.max_grad_norm,
                                    args.optim_eps, args.epochs, args.buffer_size, args.batch_size, args.target_update, preprocess_obss)
        else:
            raise ValueError("Incorrect algorithm name: {}".format(args.algo))

        if "optimizer_state" in status and status["optimizer_state"] is not None:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")
        algos.append(algo)
        G.nodes[i + 2]['state'].agent = algo

    if args.env == "MiniGrid-ConfigWorld-v0":
        anomaly_detector = BoundaryDetectorSSIM(normal_buffer_path)
        contrast_func = contrast_ssim
        contrast_value = 0.5
    elif args.env == "MiniGrid-ConfigWorld-Random":
        anomaly_detector = ClusterAnomalyDetector(normal_buffer_path)
        contrast_func = contrast_ssim
        contrast_value = 0.5

    for node in G.nodes:
        if list(G.predecessors(node)):
            if node != 0 and node != 1 and args.env != "Taxi-v0":
                # G.nodes[node]['state'].mutation = plt.imread(task_path + "/mutation" + str(node) + ".bmp")
                G.nodes[node]['state'].mutation = cv2.imread(task_path + "/mutation" + str(node) + ".bmp", cv2.IMREAD_GRAYSCALE)

    test_maps_num = 50
    single_test_num = 3
    test_logs = {"num_frames_per_episode": [], "return_per_episode": [], "env_num": []}
    log_episode_return = torch.zeros(1, device=device)
    log_episode_num_frames = torch.zeros(1, device=device) 
    log_done_counter = 0
    csv_writer.writerow(["env_num", "episode_num", "return", "num_frames", "current_state"])
    for i in range(test_maps_num):
        envs[0].fixed_map = i
        for j in range(single_test_num):
            envs[0].reset()
            mutation_buffer = []
            for node, state in G.nodes(data=True):
                # print(stateNode['state'].agent)
                if state['state'].mutation is not None:
                    mutation_buffer.append((node, state['state'].mutation))
            episode_return, episode_num_frames, current_state, stop_env, stop_obss = test_once(G, envs[0], mutation_buffer, start_node, max_steps=512, env_key="MiniGrid-ConfigWorld-v0", anomaly_detector=anomaly_detector, preprocess_obss=preprocess_obss, args=args)
            print("Env_num: ", i, "Return: ", episode_return, "Num_frames: ", episode_num_frames, "Current state: ", current_state)
            test_logs["num_frames_per_episode"].append(episode_num_frames)
            test_logs["return_per_episode"].append(episode_return)
            log_episode_return += episode_return
            log_episode_num_frames += episode_num_frames
            log_done_counter += 1
            csv_writer.writerow([i, j, episode_return, episode_num_frames, current_state])

if __name__ == "__main__":
    main()