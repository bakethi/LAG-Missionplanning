import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.JSBSim.envs import ReachBaseEnv
from algorithms.ppo.ppo_actor import PPOActor
import logging, random
logging.basicConfig(level=logging.DEBUG)

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True

def _t2n(x):
    return x.detach().cpu().numpy()

# === CONFIG ===
render = True
policy_index = "latest"
run_dir = "../scripts/results/ReachBase/1/reachbase/ppo/v1/latest"
experiment_name = run_dir.split('/')[-4]
env = ReachBaseEnv("1/reachbase")
env.seed(random.randint(0, 5))

# === POLICY ===
args = Args()
policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
policy.eval()
policy.load_state_dict(torch.load(f"{run_dir}/actor_{policy_index}.pt"))

# === INITIALIZE ENV ===
obs = env.reset()
if render:
    env.render(mode='txt', filepath=f"{experiment_name}.txt.acmi")

rnn_states = np.zeros((1, 1, args.recurrent_hidden_size), dtype=np.float32)
masks = np.ones((1, 1), dtype=np.float32)
episode_reward = 0

# === MAIN LOOP ===
while True:
    action, _, rnn_states = policy(obs, rnn_states, masks, deterministic=True)
    action = _t2n(action)
    rnn_states = _t2n(rnn_states)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    if render:
        env.render(mode='txt', filepath=f"{experiment_name}.txt.acmi")
    if done.all():
        print(info)
        break

print("Total Episode Reward:", episode_reward)
