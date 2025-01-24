import argparse
import numpy

import utils
from utils import *
from model import ACModel

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=3,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent
obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)
model_dir = utils.get_model_dir(args.model)
status=utils.get_status(model_dir)
agent_num=status["agent_num"]
# Load model
acmodels = []
for i in range(agent_num):
    acmodel = ACModel(obs_space, env.action_space, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"][i])
    acmodel.to(device)
    acmodels.append(acmodel)
# agent = utils.Agent(env.observation_space, env.action_space, model_dir,
#                     argmax=args.argmax, use_memory=args.memory, use_text=args.text)

print("Agent loaded\n")

# Run the agent
gif=False
if gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render()

for episode in range(args.episodes):
    obs, _ = env.reset()

    while True:
        env.render()
        if gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))

        preprocessed_obs = preprocess_obss([obs], device=device)
        current_state = env.Current_state()
        print(current_state)
        agent=acmodels[Choose_agent(current_state)]
        with torch.no_grad():
            dist, value = agent(preprocessed_obs)
        action = dist.sample()
        while action.item()==4:
            action = dist.sample()
        obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        done = terminated | truncated


        if done:
            break

if gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), ".gif", fps=1/args.pause)
    print("Done.")
