import argparse
import pdb
from typing import Optional
import yaml
import time
import copy
from collections import deque
import datetime
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

from graph_test import test, ddm_decision
from utils.anomaly import BoundaryDetector, BoundaryDetectorSSIM, ClusterAnomalyDetector
import math

parser = argparse.ArgumentParser()
parser.add_argument("--task-config", required=True,
                    help="the task config to use, including the graph(knowledge)")
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--fixed-map", required=True, type=int,
                    help="fixed map num(1 -50) used to finetune.")
parser.add_argument("--model", required=True, type=str,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--curriculum", default=3, type=int,
                    help="Curriculum number(1, 2, 3), used in random env")
parser.add_argument("--env", required=False, type=str, default="MiniGrid-ConfigWorld-v0",
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--test-interval", type=int, default=10,
                    help="number of updates between two tests (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--configmap", default="random_easy_small_maps.config", type=str,
                    help="the name of the map config file.")
parser.add_argument("--contrast", type=str, default="SSIM", help="Contrast function: 1. SSIM 2. HIST")
# parser.add_argument("--is-random", type=int,
#                     default=0, help="if the env is random env")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=32,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=128,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=256,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
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


G = nx.DiGraph()
args = parser.parse_args()
test_logs = {"num_frames_per_episode": [], "return_per_episode": []}

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

class RewardQueue():
    def __init__(self, csv_logger, csv_file):
        self.reward_queue = deque(maxlen=100)
        self.step_queue = deque(maxlen=100)
        self.state_queue = deque(maxlen=100)
        self.base_steps = 0
        self.csv_logger = csv_logger    
        self.csv_file = csv_file
    
    def update_reward(self, reward, extra_steps, state):
        self.reward_queue.append(reward)
        self.step_queue.append(extra_steps)
        self.state_queue.append(state)
        success_rate = self._compute_success_rate()
        self.csv_logger.writerow([self.base_steps+extra_steps, reward, state, success_rate])

    def update_base_steps(self, steps):
        self.base_steps += steps
        self.csv_file.flush()
    
    def _compute_success_rate(self):
        return sum(1 for reward in self.reward_queue if reward > 0) / len(self.reward_queue)  


def obs_To_mutation(pre_obs, obs, preprocess_obss):
    pre_image_data=preprocess_obss([pre_obs], device=device).image
    image_data=preprocess_obss([obs], device=device).image
    input_tensor = image_data - pre_image_data
    input_tensor = numpy.squeeze(input_tensor)
    return input_tensor

def calculate_epsilon(num_frames, initial_num_frames, total_frames):
    progress = (num_frames - initial_num_frames) / (total_frames - initial_num_frames)

    if progress >= 0.9:
        progress = 0.9

    epsilon = 1 - progress ** 0.5

    return epsilon


def main():
    print("args", args)
    # task_path: the path of the last task, the last folder name is "task"+task_number.
    # task_config_path: the path of the last task config(a yaml file).
    # new_task_path: the path of the new task, the last folder name is "task"+new_task_number.
    normal_buffer_path = "config/" + args.model + "/buffer/"

    state_img_path = "config/" + args.model + "/"
    task_path = os.path.join("config", args.model, args.task_config)
    task_config_path = os.path.join("config", args.model, args.task_config, "config.yaml")
    with open(task_config_path, "r") as file:
        task_config = yaml.safe_load(file)
    # get the graph structure
    for node_id in task_config['graph']['nodes']:
        G.add_node(node_id, state=stateNode(node_id, agent=None, mutation=None))
    for edge in task_config['graph']['edges']:
        G.add_edge(edge["from"], edge["to"])
    start_node = task_config['graph']['start_node']

    assert args.model is not None
    model_name = args.model 
    model_dir = utils.get_model_dir(model_name)

    finetune_model_name = model_name + f"-finetune-on-{args.fixed_map}"
    finetune_model_dir = os.path.join(model_dir, finetune_model_name)
    if not os.path.exists(finetune_model_dir):
        os.makedirs(finetune_model_dir)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(finetune_model_dir)
    csv_file, csv_logger = utils.get_csv_logger(finetune_model_dir)
    csv_episode_file, csv_episode_logger = utils.get_csv_episode_logger(finetune_model_dir)

    queue_csv_file, queue_csv_logger = utils.get_csv_logger(os.path.join(finetune_model_dir, "queue"))
    reward_queue = RewardQueue(queue_csv_logger, queue_csv_file)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)
    print("Seed:", args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load environments
    envs = []
    initial_img = None
    for i in range(args.procs):
        # kwargs = {"curriculum": args.curriculum}
        env=utils.make_env(args.env, args.seed + 10000 * i, curriculum=args.curriculum, config_path=args.configmap, fixed_map=args.fixed_map)
        initial_img, _ = env.reset()
        envs.append(env)
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    status["num_frames"] = 0
    status["update"] = 0
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    agent_num = task_config['agent_num']
    acmodels=[]
    for i in range(agent_num):
        if args.algo == "a2c" or args.algo == "ppo":
            acmodel = ACModel(obs_space, envs[0].action_space, args.text)
        elif args.algo == "dqn":
            acmodel = QNet(obs_space, envs[0].action_space, args.text)
        if "model_state" in status and status["model_state"][i] is not None:
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
        print("G.nodes[i + 2]", G.nodes[i + 2])
        G.nodes[i + 2]['state'].agent = algo

    initial_agent_num = agent_num
    if initial_agent_num > 0:
        for i in range(2, initial_agent_num + 2):
            if i < len(algos):
                print(f"Setting trained flag for algo {i}")
                if args.algo == "dqn" and hasattr(algos[i], 'trained'):
                    algos[i].trained = True
    # AnomalyNN = CNN(num_classes=2)
    # try:
    #     # AnomalyNN.load_state_dict(torch.load(AnomalyNN_model_dir))
    #     AnomalyNN = torch.load(AnomalyNN_model_dir)
    #     AnomalyNN.to(device)
    #     # print(AnomalyNN(preprocess_obss(initial_img)))
    # except OSError:
    #     AnomalyNN = lambda x: [[1.0, 0]]

    if args.env == "Taxi-v0" or args.configmap == "test_random_big_maps.config":
        anomaly_detector = ClusterAnomalyDetector(normal_buffer_path)
        contrast_func = contrast_ssim
        contrast_value = 0.5
    elif args.contrast == "HIST":
        anomaly_detector = BoundaryDetector(normal_buffer_path)
        contrast_func = contrast_hist
        contrast_value = 0.99999
    else:
        anomaly_detector = BoundaryDetectorSSIM(normal_buffer_path)
        contrast_func = contrast_ssim
        contrast_value = 0.5

    # load the mutations
    for node in G.nodes:
        if list(G.predecessors(node)):
            if node != 0 and node != 1 and args.env != "Taxi-v0":
                # G.nodes[node]['state'].mutation = plt.imread(task_path + "/mutation" + str(node) + ".bmp")
                G.nodes[node]['state'].mutation = cv2.imread(task_path + "/mutation" + str(node) + ".bmp", cv2.IMREAD_GRAYSCALE)
        if node != 0 and node != 1 and args.env != "Taxi-v0":
            G.nodes[node]['state'].env_image = plt.imread(state_img_path + "/state" + str(node) + ".bmp")

    print(G.nodes)
    print(G.edges)

    # train the model.
    num_frames = status['num_frames']
    update = status["update"]
    start_time = time.time()

    # the_max_return = agent_num.copy()
    print("G.nodes", G.nodes)
    print("G.edges", G.edges)
    # pdb.set_trace()
    no_csv_head = True
    best_test_return = float('-inf')
    best_model_states = None
    initial_num_frames = num_frames
    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        envs[0].reset()
        # ini_agent
        if args.algo == "dqn":
            epsilon =  calculate_epsilon(num_frames, initial_num_frames, args.frames)
        # print("num_frames", num_frames, "initial_num_frames", initial_num_frames, "args.frames", args.frames)
        # print("epsilon", epsilon)
        if args.algo == "a2c" or args.algo == "ppo":
            exps_list, logs1 = Mutiagent_collect_experiences(env=envs[0],
                                                            algos=algos,
                                                            contrast=contrast_func,
                                                            G=G,
                                                            device=device,
                                                            start_node=start_node,
                                                            anomaly_detector=anomaly_detector,
                                                            num_frames_per_proc=args.frames_per_proc * agent_num,
                                                            discount=args.discount,
                                                            gae_lambda=args.gae_lambda,
                                                            preprocess_obss=preprocess_obss,
                                                            discover=0,
                                                            reward_queue=reward_queue)
        elif args.algo == "dqn":
            exps_list, logs1 = Mutiagent_collect_experiences_q(env=envs[0],
                                                            algos=algos,
                                                            contrast=contrast_func,
                                                            G=G,
                                                            device=device,
                                                            start_node=start_node,
                                                            anomaly_detector=anomaly_detector,
                                                            num_frames_per_proc=args.frames_per_proc * agent_num,
                                                            preprocess_obss=preprocess_obss,
                                                            epsilon=epsilon,
                                                            discover=0)
        logs2_list = [None] * (agent_num + 2)
        # print("initial_agent_num", initial_agent_num,"agent_num", agent_num)
        # for i in range(0,len(exps_list)):
        #     print(i, len(exps_list[i].obs))
        if args.algo == "ppo":
            initial_agent_num = 0
        for i in range(initial_agent_num + 2, agent_num + 2):  
            if len(exps_list[i].obs):
                logs2 = algos[i].update_parameters(exps_list[i])
                logs2_list[i] = logs2
        logs2 = {}
        if args.algo == "a2c" or args.algo == "ppo":
            entropy_list = [None] * (agent_num + 2)
            value_list = [None] * (agent_num + 2)
            policy_loss_list = [None] * (agent_num + 2)
            value_loss_list = [None] * (agent_num + 2)
            grad_norm_list = [None] * (agent_num + 2)

            for i in range(initial_agent_num + 2, agent_num + 2):
                if len(exps_list[i].obs):
                    entropy_list[i] = logs2_list[i]["entropy"]
                    value_list[i] = logs2_list[i]["value"]
                    policy_loss_list[i] = logs2_list[i]["policy_loss"]
                    value_loss_list[i] = logs2_list[i]["value_loss"]
                    grad_norm_list[i] = logs2_list[i]["grad_norm"]
            logs2 = {
                "entropy": entropy_list,
                "value": value_list,
                "policy_loss": policy_loss_list,
                "value_loss": value_loss_list,
                "grad_norm": grad_norm_list
            }
        elif args.algo == "dqn":
            loss_list = [None] * (agent_num + 2)
            q_value_list = [None] * (agent_num + 2)
            grad_norm_list = [None] * (agent_num + 2)

            for i in range(initial_agent_num + 2, agent_num + 2):
                if len(exps_list[i].obs):
                    loss_list[i] = logs2_list[i]["loss"]
                    q_value_list[i] = logs2_list[i]["q_value"]
                    grad_norm_list[i] = logs2_list[i]["grad_norm"]
            logs2 = {
                "loss": loss_list,
                "grad_norm": grad_norm_list,
                "q_value": q_value_list
            }
        logs = {**logs1, **logs2}

        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)

            duration = int(time.time() - start_time)
            # print(logs['return_per_episode'])
            # print(logs['reshaped_return_per_episode'])
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            # print(rreturn_per_episode, "rreturn_per_episode", return_per_episode, "return_per_episode")
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            header += ["return_" + key for key in return_per_episode.keys()]
            data += rreturn_per_episode.values()
            data += return_per_episode.values()
            # header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            # data += num_frames_per_episode.values()
            if args.algo == "a2c" or args.algo == "ppo":
                header += ["policy_loss", "value_loss"]
                data += [['{:.3f}'.format(item) if item is not None else 'None' for item in logs["policy_loss"]],
                         ['{:.3f}'.format(item) if item is not None else 'None' for item in logs["value_loss"]]]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | policy_loss {} "
                    "| value_loss {}".format(*data))
                agent1_data = [logs["entropy"][2], logs["value"][2], logs["policy_loss"][2], logs["value_loss"][2], logs["grad_norm"][2]]
            elif args.algo == "dqn":
                header += ["loss", "q_value"]
                data += [['{:.3f}'.format(item) if item is not None else 'None' for item in logs["loss"]],
                         ['{:.3f}'.format(item) if item is not None else 'None' for item in logs["q_value"]]]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | loss {} "
                    "| q_value {}".format(*data))
                agent1_data = [logs["loss"][2], logs["q_value"][2], logs["grad_norm"][2]]
            header += ["agent1_entropy", "agent1_value", "agent1_policy_loss", "agent1_value_loss", "agent1_grad_norm"]

            if status["num_frames"] == 0 and no_csv_head:
                csv_logger.writerow(header)
                no_csv_head = False
            csv_logger.writerow(data + agent1_data)
            csv_file.flush()
            for episode in range(len(logs["reshaped_return_per_episode"])):
                csv_episode_logger.writerow([logs["reshaped_return_per_episode"][episode], logs["num_frames_per_episode"][episode]])
            csv_episode_file.flush()

            # for field, value in zip(header, data):
            #     tb_writer.add_scalar(field, value, num_frames)
        if args.test_interval > 0 and update % args.test_interval == 0:
            test_return_per_episode, test_num_frames_per_episode, _, _, _ = test(G, envs[0], start_node, 10, 256, args.env, preprocess_obss, anomaly_detector=anomaly_detector, args=args)
            txt_logger.info("U {} | Test reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | Test num frames:μσmM {:.2f} {:.2f} {:.2f} {:.2f}"
                            .format(10, *(test_return_per_episode.values()), *(test_num_frames_per_episode.values())))
            current_test_return = test_return_per_episode['mean']
            print("current_test_return", current_test_return)
            print("best_test_return", best_test_return)
            if current_test_return >= best_test_return:
                best_test_return = current_test_return
                best_model_states = [acmodel.state_dict() for acmodel in acmodels]

                best_status = {
                    "num_frames": num_frames,
                    "update": update,
                    "agent_num": agent_num,
                    "model_state": best_model_states,
                    "optimizer_state": algo.optimizer.state_dict(),
                    "best_test_return": best_test_return
                }
                utils.save_status(best_status, os.path.join(finetune_model_dir, "best_model"))
                txt_logger.info(f"New best model saved with test return: {best_test_return:.2f}")

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update, "agent_num": agent_num,
                      "model_state": [acmodels[i].state_dict() for i in range(agent_num)],
                      "optimizer_state": algo.optimizer.state_dict()}
            # if hasattr(preprocess_obss, "vocab"):
            #     status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, finetune_model_dir)
            txt_logger.info("Status saved")
    # save
    # status = {"num_frames": num_frames, "update": update, "agent_num": agent_num,
    #             "model_state": [acmodels[i].state_dict() for i in range(agent_num)],
    #             "optimizer_state": algo.optimizer.state_dict()}
    # # if hasattr(preprocess_obss, "vocab"):
    # #     status["vocab"] = preprocess_obss.vocab.vocab
    # utils.save_status(status, model_dir)
    # txt_logger.info("Status saved")

    for i, acmodel in enumerate(acmodels):
        acmodel.load_state_dict(best_model_states[i])
    txt_logger.info(f"Loaded best model with test return: {best_test_return:.2f}")
    model_state = [acmodel.state_dict() for acmodel in acmodels]
    optimizer_state = algo.optimizer.state_dict()
    if return_per_episode['mean'] <= 0.5:
        print("No save bad model.")
        model_state[-1] = None
        optimizer_state = None
        print(len(model_state))
    final_status = {
        "num_frames": num_frames,
        "update": update,
        "agent_num": agent_num,
        "model_state": model_state, #[acmodel.state_dict() for acmodel in acmodels],
        "optimizer_state": optimizer_state,
        "best_test_return": best_test_return
    }
    utils.save_status(final_status, finetune_model_dir)
    txt_logger.info("Final status saved")

    if return_per_episode['mean'] <= 0.5:
        return "fail train"
    else:
        return "successfull train"
    # random discover, save the changes.
    # until get a familiar change(state)
    # change the graph

if __name__ == "__main__":
    ret_state = main()
    ret_value = 0
    if ret_state == "can_solve":
        ret_value = 0
    elif ret_state == "fail to discover anomaly":
        ret_value = 1
    elif ret_state == "fail train":
        ret_value = 2
    elif ret_state == "successfull train":
        ret_value = 3
    print(f"Return value: {ret_value}")
    sys.exit(ret_value)
# return "can_solve"
# return "fail to discover anomaly"
# return "fail train"
# return "successfull train"
