import networkx as nx
import cv2
import torch
from torchvision import transforms
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
from torch.distributions.categorical import Categorical
import sys
import heapq
import copy
sys.path.append("..")
import numpy
import random
from .abl_trace import abl_trace
from .env import copy_env
import matplotlib.pyplot as plt
from torchvision.utils import save_image
# from skimage.metrics import structural_similarity as ssim
# from .process import contrast_ssim  

def get_state(env):
    return env.Current_state()

def sample_from_selected_dimensions(logits, selected_dims):
    mask = torch.full_like(logits, fill_value=-float('Inf'))
    mask[:, selected_dims] = 0
    masked_logits = logits + mask
    probabilities = torch.softmax(masked_logits, dim=1)
    samples = torch.multinomial(probabilities, num_samples=1)
    return samples

def RGB2GARY_ROI(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processed = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray_image[y:y + h, x:x + w]
        processed.append(roi)
    
    return gray_image, processed

def obs_To_state(current_state,
                preprocess_obss,
                anomaly_detector, 
                contrast, 
                G: nx.DiGraph,
                pre_obs,
                obs,
                device,
                is_add_normal_samples=False):
    # print("current_state", current_state)
    pre_image_data=preprocess_obss([pre_obs], device=device)
    image_data=preprocess_obss([obs], device=device)
    input_tensor = image_data.image[0]-pre_image_data.image[0]
    mutation = numpy.squeeze(input_tensor).cpu().numpy().astype(numpy.uint8)
    _, roi_list = RGB2GARY_ROI(mutation)
    is_anomaly = False
    for roi in roi_list:
        if not is_add_normal_samples:
            break
        is_anomaly = anomaly_detector.is_known_roi(roi, add_to_buffer=is_add_normal_samples)
        if is_anomaly:
            break

    # if is_add_normal_samples:
    #     anomaly_detector.add_normal_samples(mutation)
    # anomaly_mutation = transforms.ToTensor()(mutation).cuda().unsqueeze(0)  
    # if anomalyNN(anomaly_mutation)[0, 0] >= anomalyNN(anomaly_mutation)[0, 1]:
    # if is_add_normal_samples or not is_anomaly:
    if is_add_normal_samples:
        return current_state

    # if not anomaly_detector.detect_anomaly(mutation):
    similiarity = []
    for next_state in list(G.successors(current_state)):
        # print("next_state", next_state)
        # print("G.nodes[next_state]['state'].mutation", G.nodes[next_state]['state'].mutation)
        for roi in roi_list:
            similiarity.append((next_state, anomaly_detector.contrast(roi, G.nodes[next_state]['state'].mutation)))
            # print(contrast_ssim(roi, G.nodes[next_state]['state'].mutation))

        # print(contrast(mutation, G.nodes[next_state]['state'].mutation))
    # print("similiarity", similiarity)
    if len(similiarity) != 0:
        output = max(similiarity, key=lambda x: x[1]) 
        # if output[1] != 0:
        #     print("output", output)
    else:
        output = (current_state, 0)
    # plt.imshow(mutation)
    # plt.show()
    if output[1] < anomaly_detector.contrast_value:
        # print("no", output[1])
        return current_state
    # print("yes", output[0], output[1])
    output = output[0]
    # print(output)
    # plt.imshow(mutation)
    # plt.show()
    return output

def Mutiagent_collect_experiences(env, 
                                algos,
                                contrast,
                                G: nx.DiGraph,
                                start_node,
                                anomaly_detector,  
                                device,
                                num_frames_per_proc, 
                                discount, 
                                gae_lambda, 
                                preprocess_obss,
                                discover,
                                reward_queue=None,):

    is_add_normal_samples = False
    if len(G.nodes) <= 3 and not discover:
        is_add_normal_samples = True

    # def pre_obs_softmax(model, obs):
    #     image_data=preprocess_obss([obs], device=device)
    #     input_tensor = image_data.image[0]
    #     input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    #     output = model(input_batch)
    #     prob = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy()[0]
    #     # print("prob", prob)
    #     return prob

    agent_num=len(algos)
    obs=env.gen_obs()
    pre_obs=obs

    mask_trace=[]
    ###
    state_mask_trace=[]
    obs_trace=[]
    state_trace=[]
    action_trace=[]
    value_trace=[]
    reward_trace=[]
    log_prob_trace=[]

    mask=torch.ones(1, device=device)
    state_mask = torch.ones(1, device=device)
    done=0

    done_counter=0
    log_return=[0]
    log_num_frames=[0]
    episode_reward_return = torch.zeros(1, device=device)
    log_episode_num_frames = torch.zeros(1, device=device)


    env_ini_state = start_node 
    current_state=env_ini_state
    state_ini_flag=False

    for step_num in range(num_frames_per_proc):

        preprocessed_obs = preprocess_obss([obs], device=device)

        # t,prob_dist=obs_To_state(StateNN, pre_obs, obs)

        # if t==0:
        #     current_state=current_state
        # else:
        #     candidate_list=Candidate(current_state)
        #     t=sample_from_selected_dimensions(prob_dist,candidate_list)
        #     current_state = t
        if not state_ini_flag:
            current_state = obs_To_state(current_state,
                                     preprocess_obss, 
                                     anomaly_detector, 
                                     contrast, 
                                     G, 
                                     pre_obs, 
                                     obs, 
                                     device,
                                     is_add_normal_samples)
        else:
            current_state=env_ini_state
            state_ini_flag=False

        if current_state != 0 and current_state != 1:
            agent = G.nodes[current_state]['state'].agent
            with torch.no_grad():
                dist, value = agent.acmodel(preprocessed_obs)
            action = dist.sample()

        if done:
            if reward > 0:
                current_state = 1
            if reward < 0:
                current_state = 0
            ######
            if reward_queue is not None:
                reward_queue.update_reward(reward, step_num, current_state)
            env.reset()
            next_obs=env.gen_obs()
            # plt.imshow(next_obs['image'].astype(numpy.uint8))
            # plt.show()
            reward=0
            terminated=0
            truncated=0
            state_ini_flag=True
        else:
            next_obs, reward, terminated, truncated, _ = env.step(int(action.cpu().numpy()))

        state_trace.append(current_state)
        mask_trace.append(mask)
        state_mask_trace.append(state_mask)
        obs_trace.append(obs)
        action_trace.append(action)
        value_trace.append(value)
        reward_trace.append(torch.tensor(reward, device=device,dtype=torch.float))
        log_prob_trace.append(dist.log_prob(action))

#####################################################
        done = terminated|truncated
        mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
        ####
        change_state = len(state_trace) >= 2 and (done or current_state != state_trace[-2])
        state_mask = 1 - torch.tensor(change_state, device=device, dtype=torch.float)
        
        pre_obs=obs
        obs=next_obs

        episode_reward_return += torch.tensor(reward, device=device, dtype=torch.float)
        log_episode_num_frames += torch.tensor(1, device=device)

        if done:
            done_counter += 1
            log_return.append(episode_reward_return.item())
            log_num_frames.append(log_episode_num_frames.item())

        episode_reward_return *= mask
        log_episode_num_frames *= mask

        torch.cuda.empty_cache()

    if reward_queue is not None:
        reward_queue.update_base_steps(num_frames_per_proc)

    keep1 = max(done_counter,1)
    log1 = {
        "return_per_episode": log_return[-keep1:],
        "num_frames_per_episode": log_num_frames[-keep1:],
        "num_frames": num_frames_per_proc
    }


    for i in range(len(state_trace) - 1):
        if state_trace[i] > state_trace[i + 1]:
            # mental reward
            if reward_trace[i] == 0 and state_trace[i] != 0 and state_trace[i] != 1:
                # print("State change:", state_trace[i], " ", state_trace[i+1], " ", reward_trace[i], "", reward_trace[i+1])
                reward_trace[i] = torch.tensor(1, device=device, dtype=torch.float)

    next_value=value_trace[-1]
    advantage_trace=[0]*len(action_trace)
    for i in reversed(range(num_frames_per_proc)):
        next_mask = state_mask_trace[i + 1] if i < num_frames_per_proc - 1 else state_mask
        next_value = value_trace[i + 1] if i < num_frames_per_proc - 1 else next_value
        next_advantage = advantage_trace[i + 1] if i < num_frames_per_proc - 1 else 0

        delta = reward_trace[i] + discount * next_value * next_mask - value_trace[i]
        advantage_trace[i] = delta + discount * gae_lambda * next_advantage * next_mask

    # print("After abl", state_trace)
    exps_list=[]

    for i in range(agent_num + 2):
        exps=DictList()
        exps.obs=[]
        exps.action=[]
        exps.reward=[]
        exps.value=[]
        exps.advantage=[]
        exps.log_prob=[]
        exps_list.append(exps)

    # print(state_trace)
    # print([int(i.item()) for i in mask_trace])
    start_index=0
    for i in range(len(state_trace)-1):
        if state_trace[i]!=state_trace[i+1]:
            id=state_trace[start_index]
            # mental reward
            if id!=0 and id!=1:
                exps_list[id].obs.extend(obs_trace[start_index:i+1])
                exps_list[id].action.extend(action_trace[start_index:i+1])
                exps_list[id].reward.extend(reward_trace[start_index:i+1])
                exps_list[id].value.extend(value_trace[start_index:i+1])
                exps_list[id].advantage.extend(advantage_trace[start_index:i+1])
                exps_list[id].log_prob.extend(log_prob_trace[start_index:i+1])
            start_index=i+1
    if start_index<len(state_trace)-2:
        id = state_trace[start_index]
        exps_list[id].obs.extend(obs_trace[start_index:])
        exps_list[id].action.extend(action_trace[start_index:])
        exps_list[id].reward.extend(reward_trace[start_index:])
        exps_list[id].value.extend(value_trace[start_index:])
        exps_list[id].advantage.extend(advantage_trace[start_index:])
        exps_list[id].log_prob.extend(log_prob_trace[start_index:])

    for i in range(agent_num):
        exp_len=len(exps_list[i].obs)
        if exp_len:
            exps_list[i].obs = preprocess_obss(exps_list[i].obs, device=device)
            exps_list[i].action = torch.tensor(exps_list[i].action, device=device, dtype=torch.int)
            exps_list[i].reward = torch.tensor(exps_list[i].reward, device=device)
            exps_list[i].value = torch.tensor(exps_list[i].value, device=device)
            exps_list[i].advantage = torch.tensor(exps_list[i].advantage, device=device)
            exps_list[i].log_prob = torch.tensor(exps_list[i].log_prob, device=device)
            exps_list[i].returnn = exps_list[i].value + exps_list[i].advantage
    # print([int(i.item()) for i in reward_trace])
    log_reshaped_return=[0]
    log_done_counter=0
    log_episode_reshaped_return=torch.zeros(1, device=device)
    for i in range(len(reward_trace)):
        log_episode_reshaped_return+=reward_trace[i]
        ## 1111修改：去除此处的未完成episode的经验
        # if mask_trace[i].item() == 0 or i==len(reward_trace)-1:
        if mask_trace[i].item() == 0:
            log_done_counter += 1
            log_reshaped_return.append(log_episode_reshaped_return.item())
        log_episode_reshaped_return *= mask_trace[i]

    keep2 = max(log_done_counter, 1)
    log2={
                "reshaped_return_per_episode": log_reshaped_return[-keep2:],
            }

    return exps_list, {**log1, **log2}

def Mutiagent_collect_experiences_q(env, 
                                algos,
                                contrast,
                                G: nx.DiGraph,
                                start_node,
                                anomaly_detector,  
                                device,
                                num_frames_per_proc, 
                                preprocess_obss,
                                epsilon,
                                discover):
    # 这里是指要不要训练异常检测器，如果节点数小于等于3，就是初始状态，训练异常检测器。
    is_add_normal_samples = False
    if len(G.nodes) <= 3 and not discover:
        is_add_normal_samples = True
    # def pre_obs_softmax(model, obs):
    #     image_data=preprocess_obss([obs], device=device)
    #     input_tensor = image_data.image[0]
    #     input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    #     output = model(input_batch)
    #     prob = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy()[0]
    #     # print("prob", prob)
    #     return prob

    agent_num=len(algos)
    obs=env.gen_obs()
    pre_obs=obs

    mask_trace=[]
    obs_trace=[]
    state_trace=[]
    action_trace=[]
    q_value_trace=[]
    reward_trace=[]
    next_obs_trace = []

    mask=torch.ones(1, device=device)
    done=0

    done_counter=0
    log_return=[0]
    log_num_frames=[0]
    episode_reward_return = torch.zeros(1, device=device)
    log_episode_num_frames = torch.zeros(1, device=device)


    env_ini_state = start_node 
    current_state=env_ini_state
    state_ini_flag=False

    for _ in range(num_frames_per_proc):

        preprocessed_obs = preprocess_obss([obs], device=device)

        # t,prob_dist=obs_To_state(StateNN, pre_obs, obs)

        # if t==0:
        #     current_state=current_state
        # else:
        #     candidate_list=Candidate(current_state)
        #     t=sample_from_selected_dimensions(prob_dist,candidate_list)
        #     current_state = t
        if not state_ini_flag:
            current_state = obs_To_state(current_state,
                                        preprocess_obss, 
                                        anomaly_detector, 
                                        contrast, 
                                        G, 
                                        pre_obs, 
                                        obs, 
                                        device,
                                        is_add_normal_samples)
        else:
            current_state=env_ini_state
            state_ini_flag=False

        if current_state != 0 and current_state != 1:
            agent = G.nodes[current_state]['state'].agent

            with torch.no_grad():
                q_values = agent.acmodel(preprocessed_obs)
                if agent.trained == False:
                    action = agent.select_action(preprocessed_obs, epsilon)
                else:
                    action = agent.select_action(preprocessed_obs, epsilon/10)
        # print("output_state", current_state)
        if done:
            # print("done")
            if reward > 0:
                current_state = 1
            if reward < 0:
                current_state = 0
            env.reset()
            next_obs=env.gen_obs()
            reward=0
            terminated=0
            truncated=0
            state_ini_flag=True
        else:
            next_obs, reward, terminated, truncated, _ = env.step(int(action.cpu().numpy()))
        # print("reward", reward)
        # print("terminated", terminated)
        # print("truncated", truncated)
        state_trace.append(current_state)
        mask_trace.append(mask)
        obs_trace.append(obs)
        action_trace.append(action)
        q_value_trace.append(q_values)
        reward_trace.append(torch.tensor(reward, device=device,dtype=torch.float))
        next_obs_trace.append(next_obs)

#####################################################
        done = terminated|truncated
        mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
        pre_obs=obs
        obs=next_obs

        episode_reward_return += torch.tensor(reward, device=device, dtype=torch.float)
        log_episode_num_frames += torch.tensor(1, device=device)

        if done:
            done_counter += 1
            log_return.append(episode_reward_return.item())
            log_num_frames.append(log_episode_num_frames.item())

        episode_reward_return *= mask
        log_episode_num_frames *= mask

        # 在不再需要时删除变量
        del preprocessed_obs, next_obs

        # 清除未使用的显存
        torch.cuda.empty_cache()

    keep1 = max(done_counter,1)
    log1 = {
        "return_per_episode": log_return[-keep1:],
        "num_frames_per_episode": log_num_frames[-keep1:],
        "num_frames": num_frames_per_proc
    }

    for i in range(len(state_trace) - 1):
        if state_trace[i] > state_trace[i + 1]:
            # mental reward
            if reward_trace[i] == 0 and state_trace[i] != 0 and state_trace[i] != 1:
                reward_trace[i] = torch.tensor(1, device=device, dtype=torch.float)

    # print("After abl", state_trace)
    exps_list=[]

    for i in range(agent_num):
        exps=DictList()
        exps.obs=[]
        exps.action=[]
        exps.reward=[]
        exps.mask = []
        exps.obs_ = []
        exps_list.append(exps)

    # print(state_trace)
    # print([int(i.item()) for i in mask_trace])
    start_index=0
    for i in range(len(state_trace)-1):
        if state_trace[i]!=state_trace[i+1]:
            id=state_trace[start_index]
            # mental reward
            if id!=0 and id!=1:
                exps_list[id].obs.extend(obs_trace[start_index:i+1])
                exps_list[id].action.extend(action_trace[start_index:i+1])
                exps_list[id].reward.extend(reward_trace[start_index:i+1])
                exps_list[id].mask.extend(mask_trace[start_index:i + 1])
                exps_list[id].obs_.extend(next_obs_trace[start_index:i + 1])
            start_index=i+1
    if start_index<len(state_trace)-2:
        id = state_trace[start_index]
        exps_list[id].obs.extend(obs_trace[start_index:])
        exps_list[id].action.extend(action_trace[start_index:])
        exps_list[id].reward.extend(reward_trace[start_index:])
        exps_list[id].mask.extend(mask_trace[start_index:])
        exps_list[id].obs_.extend(next_obs_trace[start_index:])

    for i in range(agent_num):
        exp_len=len(exps_list[i].obs)
        if exp_len:
            exps_list[i].obs = preprocess_obss(exps_list[i].obs, device=device)
            exps_list[i].action = torch.tensor(exps_list[i].action, device=device, dtype=torch.int)
            exps_list[i].reward = torch.tensor(exps_list[i].reward, device=device)
            exps_list[i].mask = torch.tensor(exps_list[i].mask, device=device)
            exps_list[i].obs_ = preprocess_obss(exps_list[i].obs_, device=device)
    # print([int(i.item()) for i in reward_trace])
    log_reshaped_return=[0]
    log_done_counter=0
    log_episode_reshaped_return=torch.zeros(1, device=device)
    for i in range(len(reward_trace)):
        log_episode_reshaped_return+=reward_trace[i]
        if mask_trace[i].item() == 0:
            log_done_counter += 1
            log_reshaped_return.append(log_episode_reshaped_return.item())
        log_episode_reshaped_return *= mask_trace[i]

    keep2 = max(log_done_counter, 1)
    log2={
                "reshaped_return_per_episode": log_reshaped_return[-keep2:],
            }
    # print(log_reshaped_return[-keep2:])
    # print(action_trace)
    # print([i.item() for i in mask_trace])
    # memory_tracker.log_memory('end_log_experiences')
    return exps_list, {**log1, **log2}


def collect_experiences_mutation(algo, 
                                    start_env, 
                                    get_mutation_score, 
                                    mutation_buffer, 
                                    mutation_value, 
                                    contrast,
                                    known_mutation_buffer,
                                    arrived_state_buffer,
                                    preprocess_obss,
                                    env_name,
                                    anomaly_detector,):
    """Collects rollouts and computes advantages.

    Runs several environments concurrently. The next actions are computed
    in a batch mode for all environments at the same time. The rollouts
    and advantages from all environments are concatenated together.

    Returns
    -------
    exps : DictList
        Contains actions, rewards, advantages etc as attributes.
        Each attribute, e.g. `exps.reward` has a shape
        (algo.num_frames_per_proc * num_envs, ...). k-th block
        of consecutive `algo.num_frames_per_proc` frames contains
        data obtained from the k-th environment. Be careful not to mix
        data from different environments!
    logs : dict
        Useful stats about the training process, including the average
        reward, policy loss, value loss, etc.
    """
    env = copy_env(start_env, env_name)
    trace_roi_buffer = []
    parallel_env = ParallelEnv([env])
    done = (True,)
    last_done = (True,)
    episode_return = 0
    obs = parallel_env.gen_obs()
    for i in range(algo.num_frames_per_proc):
        # Do one agent-environment interaction
        last_done = done

        preprocessed_obs = preprocess_obss(obs, device=algo.device)

        with torch.no_grad():
            if algo.acmodel.recurrent:
                dist, value, memory = algo.acmodel(preprocessed_obs, algo.memory * algo.mask.unsqueeze(1))
            else:
                dist, value = algo.acmodel(preprocessed_obs)

        x=dist.probs+dist.probs
        combined_dist = Categorical(probs=x)
        action = combined_dist.sample()
        obs, reward, terminated, truncated, _ = parallel_env.step(action.cpu().numpy())
        if reward[0] > 0:
            arrived_state_buffer.append(1)
        done = tuple(a | b for a, b in zip(terminated, truncated))
        if done[0]:
            # env = copy_env(start_env, env_name)
            env.reset()
            parallel_env = ParallelEnv([env])
            obs = parallel_env.gen_obs()

        # print(obs)
        # print(algo.obss)
        the_preprocessed_obs = preprocess_obss(obs, device=algo.device)
        # plt.imshow(numpy.squeeze(the_preprocessed_obs.image).cpu().numpy().astype(numpy.uint8))
        # plt.show()
        
        if done[0]:
            mutation = the_preprocessed_obs.image - the_preprocessed_obs.image
        else:
            mutation = the_preprocessed_obs.image - preprocessed_obs.image
        mutation = numpy.squeeze(mutation)
        mutation = mutation.cpu().numpy().astype(numpy.uint8)
        _, mutation_roi_list = RGB2GARY_ROI(mutation)
        ### 检查突变是否与已知的突变相同
        # print(mutation.shape)
        deleted_roi_index = []
        for index in range(len(mutation_roi_list)):
            for _, (idx, mutation_) in enumerate(known_mutation_buffer):
                if contrast(mutation_roi_list[index], mutation_) > 0.6:
                    arrived_state_buffer.append(idx)
                    reward = (1, )
                    done = (True,)
                    break
            if mutation_roi_list[index].shape[0] < 8 or mutation_roi_list[index].shape[1] < 8:
                deleted_roi_index.append(index)
                continue
            if get_mutation_score(mutation_roi_list[index]) < mutation_value or reward[0] != 0:
            # if reward[0] != 0:
                break
            is_in_buffer = False
            for idx, (score_, mutation_, times_, env_) in enumerate(mutation_buffer):
                if contrast(mutation_roi_list[index], mutation_) > 0.6:
                    mutation_buffer[idx] = (score_, mutation_, times_ + 1, algo.env)
                    is_in_buffer = True
                    break
            if not is_in_buffer:
                #print(get_mutation_score(mutation).dtype)
                # heapq.heappush(mutation_buffer, (get_mutation_score(mutation), mutation, 1, copy.deepcopy(algo.env)))
                mutation_buffer.append((get_mutation_score(mutation_roi_list[index]), mutation_roi_list[index], 1, algo.env))
            
        mutation_roi_list = [mutation_roi_list[index] for index in range(len(mutation_roi_list)) if index not in deleted_roi_index]

        episode_return += reward[0]
        # if env_name == "Taxi-v0" or env_name == "MiniGrid-ConfigWorld-Random":
        if not done[0]:
            trace_roi_buffer.extend(mutation_roi_list)
        else:
            print("Episode return: ", episode_return)
            if reward[0] > 0 and episode_return > 0:
                # print("Reward", reward)
                anomaly_roi = anomaly_detector.add_samples(trace_roi_buffer)
                mutation_buffer.append((get_mutation_score(anomaly_roi), anomaly_roi, 1, algo.env))
            trace_roi_buffer = []
        episode_return *= 1 - done[0]

        algo.obss[i] = algo.obs
        algo.obs = obs
        if algo.acmodel.recurrent:
            algo.memories[i] = algo.memory
            algo.memory = memory
        algo.masks[i] = algo.mask
        algo.mask = 1 - torch.tensor(done, device=algo.device, dtype=torch.float)
        algo.actions[i] = action
        algo.values[i] = value
        if algo.reshape_reward is not None:
            algo.rewards[i] = torch.tensor([
                algo.reshape_reward(obs_, action_, reward_, done_)
                for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
            ], device=algo.device)
        else:
            algo.rewards[i] = torch.tensor(reward, device=algo.device)
        algo.log_probs[i] = dist.log_prob(action)

        # Update log values

        algo.log_episode_return += torch.tensor(reward, device=algo.device, dtype=torch.float)
        algo.log_episode_reshaped_return += algo.rewards[i]
        algo.log_episode_num_frames += torch.ones(algo.num_procs, device=algo.device)

        for i, done_ in enumerate(done):
            if done_:
                algo.log_done_counter += 1
                algo.log_return.append(algo.log_episode_return[i].item())
                algo.log_reshaped_return.append(algo.log_episode_reshaped_return[i].item())
                algo.log_num_frames.append(algo.log_episode_num_frames[i].item())

        algo.log_episode_return *= algo.mask
        algo.log_episode_reshaped_return *= algo.mask
        algo.log_episode_num_frames *= algo.mask

        torch.cuda.empty_cache()

    # Add advantage and return to experiences

    preprocessed_obs = preprocess_obss(algo.obs, device=algo.device)
    with torch.no_grad():
        if algo.acmodel.recurrent:
            _, next_value, _ = algo.acmodel(preprocessed_obs, algo.memory * algo.mask.unsqueeze(1))
        else:
            _, next_value = algo.acmodel(preprocessed_obs)

    for i in reversed(range(algo.num_frames_per_proc)):
        next_mask = algo.masks[i+1] if i < algo.num_frames_per_proc - 1 else algo.mask
        next_value = algo.values[i+1] if i < algo.num_frames_per_proc - 1 else next_value
        next_advantage = algo.advantages[i+1] if i < algo.num_frames_per_proc - 1 else 0

        delta = algo.rewards[i] + algo.discount * next_value * next_mask - algo.values[i]
        algo.advantages[i] = delta + algo.discount * algo.gae_lambda * next_advantage * next_mask

    # Define experiences:
    #   the whole experience is the concatenation of the experience
    #   of each process.
    # In comments below:
    #   - T is algo.num_frames_per_proc,
    #   - P is algo.num_procs,
    #   - D is the dimensionality.
    exps = DictList()
    exps.obs = [algo.obss[i][j]
                for j in range(algo.num_procs)
                for i in range(algo.num_frames_per_proc)]
    if algo.acmodel.recurrent:
        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = algo.memories.transpose(0, 1).reshape(-1, *algo.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = algo.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
    # for all tensors below, T x P -> P x T -> P * T
    exps.action = algo.actions.transpose(0, 1).reshape(-1)
    exps.value = algo.values.transpose(0, 1).reshape(-1)
    exps.reward = algo.rewards.transpose(0, 1).reshape(-1)
    exps.advantage = algo.advantages.transpose(0, 1).reshape(-1)
    exps.returnn = exps.value + exps.advantage
    exps.log_prob = algo.log_probs.transpose(0, 1).reshape(-1)

    # Preprocess experiences
    # print("obs", exps.obs)
    exps.obs = algo.preprocess_obss(exps.obs, device=algo.device)

    # Log some values
    keep = max(algo.log_done_counter, algo.num_procs)
    logs = {
        "return_per_episode": algo.log_return[-keep:],
        "reshaped_return_per_episode": algo.log_reshaped_return[-keep:],
        "num_frames_per_episode": algo.log_num_frames[-keep:],
        "num_frames": algo.num_frames
    }

    algo.log_done_counter = 0
    algo.log_return = algo.log_return[-algo.num_procs:]
    algo.log_reshaped_return = algo.log_reshaped_return[-algo.num_procs:]
    algo.log_num_frames = algo.log_num_frames[-algo.num_procs:]

    return exps, logs

def collect_experiences_mutation_q(algo, 
                                start_env, 
                                get_mutation_score, 
                                mutation_buffer, 
                                mutation_value, 
                                contrast,
                                known_mutation_buffer,
                                arrived_state_buffer,
                                preprocess_obss,
                                env_name,
                                epsilon,
                                anomaly_detector,):
    """为DQN收集经验的版本。
    主要区别:
    1. 使用epsilon-greedy策略选择动作
    2. 收集的经验包含 (s,a,r,s',done) 元组
    """
    env = copy_env(start_env, env_name)
    parallel_env = ParallelEnv([env])
    done = (True,)
    last_done = (True,)
    trace_roi_buffer = []
    episode_return = 0
    obs = parallel_env.gen_obs()
    for i in range(algo.num_frames_per_proc):
        last_done = done
        preprocessed_obs = preprocess_obss(obs, device=algo.device)

        with torch.no_grad():
            q_values = algo.acmodel(preprocessed_obs)
            action = algo.select_action(preprocessed_obs, epsilon)
            # print("action", action) 

        obs, reward, terminated, truncated, _ = parallel_env.step(action.cpu().numpy())
        if reward[0] > 0:
            arrived_state_buffer.append(1)
            
        done = tuple(a | b for a, b in zip(terminated, truncated))
        if done[0]:
            env = copy_env(start_env, env_name)
            parallel_env = ParallelEnv([env])
            obs = parallel_env.gen_obs()

        the_preprocessed_obs = preprocess_obss(obs, device=algo.device)
        
        if done[0]:
            mutation = the_preprocessed_obs.image - the_preprocessed_obs.image
        else:
            mutation = the_preprocessed_obs.image - preprocessed_obs.image
        mutation = numpy.squeeze(mutation)
        mutation = mutation.cpu().numpy().astype(numpy.uint8)
        _, mutation_roi_list = RGB2GARY_ROI(mutation)

        deleted_roi_index = []
        for index in range(len(mutation_roi_list)):
            # if get_mutation_score(mutation_roi) < mutation_value or reward[0] != 0:
            #     continue
            for _, (idx, mutation_) in enumerate(known_mutation_buffer):
                if contrast(mutation_roi_list[index], mutation_) > 0.6:
                    arrived_state_buffer.append(idx)
                    reward = (1, )
                    done = (True,)
                    break
            if mutation_roi_list[index].shape[0] < 8 or mutation_roi_list[index].shape[1] < 8:
                deleted_roi_index.append(index)
                continue
            if get_mutation_score(mutation_roi_list[index]) < mutation_value or reward[0] != 0:
                break
            is_in_buffer = False
            for idx, (score_, mutation_, times_, env_) in enumerate(mutation_buffer):
                if contrast(mutation_roi_list[index], mutation_) > 0.6:
                    mutation_buffer[idx] = (score_, mutation_, times_ + 1, algo.env)
                    is_in_buffer = True
                    break
            if not is_in_buffer:
                mutation_buffer.append((get_mutation_score(mutation_roi_list[index]), mutation_roi_list[index], 1, algo.env))
        
        mutation_roi_list = [mutation_roi_list[index] for index in range(len(mutation_roi_list)) if index not in deleted_roi_index]

        episode_return += reward[0]
        if env_name == "Taxi-v0" or env_name == "MiniGrid-ConfigWorld-Random":
            if not done[0]:
                trace_roi_buffer.extend(mutation_roi_list)
            else:
                print("Episode return: ", episode_return)
                if reward[0] > 0 and episode_return > 0:
                    print("Reward", reward)
                    anomaly_roi = anomaly_detector.add_samples(trace_roi_buffer)
                    mutation_buffer.append((get_mutation_score(anomaly_roi), anomaly_roi, 1, algo.env))
                trace_roi_buffer = []
        episode_return *= 1 - done[0]

        # 更新经验值
        algo.obss[i] = algo.obs
        algo.obs = obs
        algo.masks[i] = algo.mask
        algo.mask = 1 - torch.tensor(done, device=algo.device, dtype=torch.float)
        algo.actions[i] = action
        if algo.reshape_reward is not None:
            algo.rewards[i] = torch.tensor([
                algo.reshape_reward(obs_, action_, reward_, done_)
                for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
            ], device=algo.device)
        else:
            algo.rewards[i] = torch.tensor(reward, device=algo.device)
        algo.obs_[i] = obs  # 存储下一个状态

        # 更新日志值
        algo.log_episode_return += torch.tensor(reward, device=algo.device, dtype=torch.float)
        algo.log_episode_reshaped_return += algo.rewards[i]
        algo.log_episode_num_frames += torch.ones(algo.num_procs, device=algo.device)

        for i, done_ in enumerate(done):
            if done_:
                algo.log_done_counter += 1
                algo.log_return.append(algo.log_episode_return[i].item())
                algo.log_reshaped_return.append(algo.log_episode_reshaped_return[i].item())
                algo.log_num_frames.append(algo.log_episode_num_frames[i].item())

        algo.log_episode_return *= algo.mask
        algo.log_episode_reshaped_return *= algo.mask
        algo.log_episode_num_frames *= algo.mask

        del preprocessed_obs, the_preprocessed_obs
        torch.cuda.empty_cache()

    # 定义经验
    exps = DictList()
    exps.obs = [algo.obss[i][j]
                for j in range(algo.num_procs)
                for i in range(algo.num_frames_per_proc)]
    exps.action = algo.actions.transpose(0, 1).reshape(-1)
    exps.reward = algo.rewards.transpose(0, 1).reshape(-1)
    exps.mask = algo.masks.transpose(0, 1).reshape(-1)
    exps.obs_ = [algo.obs_[i][j]
                 for j in range(algo.num_procs)
                 for i in range(algo.num_frames_per_proc)]

    # 预处理经验
    exps.obs = algo.preprocess_obss(exps.obs, device=algo.device)
    exps.obs_ = algo.preprocess_obss(exps.obs_, device=algo.device)

    # 记录日志
    keep = max(algo.log_done_counter, algo.num_procs)
    logs = {
        "return_per_episode": algo.log_return[-keep:],
        "reshaped_return_per_episode": algo.log_reshaped_return[-keep:],
        "num_frames_per_episode": algo.log_num_frames[-keep:],
        "num_frames": algo.num_frames
    }

    algo.log_done_counter = 0
    algo.log_return = algo.log_return[-algo.num_procs:]
    algo.log_reshaped_return = algo.log_reshaped_return[-algo.num_procs:]
    algo.log_num_frames = algo.log_num_frames[-algo.num_procs:]

    return exps, logs





