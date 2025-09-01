import numpy as np
import torch
import random
import sys, os
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.JSBSim.envs import MultiWaypointEnv
from algorithms.ppo.ppo_actor import PPOActor
import logging

# Import the heatmap class
from heatmap import FlightHeatmap

logging.basicConfig(level=logging.INFO)

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
run_dir = "../scripts/results/MultiWaypoint/1/multiwaypoint/ppo/v1/latest"
experiment_name = run_dir.split('/')[-4]

# === ENV SETUP ===
env = MultiWaypointEnv("1/multiwaypoint")
env.seed(random.randint(0, 5))

# === ACMI OUTPUT PATH SETUP ===
env_name = env.__class__.__name__  # or use env.env_name if available
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
acmi_dir = os.path.join(".", env_name, timestamp)
os.makedirs(acmi_dir, exist_ok=True)
acmi_filepath = os.path.join(acmi_dir, f"{experiment_name}.txt.acmi")

# === POLICY LOAD ===
args = Args()
policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cpu"))
policy.eval()
policy.load_state_dict(torch.load(f"{run_dir}/actor_{policy_index}.pt", map_location='cpu'))

# === EPISODE INIT ===
obs = env.reset()
if render:
    env.render(mode='txt', filepath=acmi_filepath)

rnn_states = np.zeros((1, 1, args.recurrent_hidden_size), dtype=np.float32)
masks = np.ones((1, 1), dtype=np.float32)
episode_reward = 0

# === MAIN INFERENCE LOOP ===
with tqdm(total=env.max_steps, desc="Running Simulation", unit="step", ncols=100) as pbar:
    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        masks_tensor = torch.tensor(masks, dtype=torch.float32)
        rnn_tensor = torch.tensor(rnn_states, dtype=torch.float32)

        with torch.no_grad():
            action, _, rnn_tensor = policy(obs_tensor, rnn_tensor, masks_tensor, deterministic=True)

        action = _t2n(action)
        rnn_states = _t2n(rnn_tensor)

        obs, reward, done, info = env.step(action)
        episode_reward += reward
        pbar.update(1)

        if render:
            env.render(mode='txt', filepath=acmi_filepath)

        if done.all():
            pbar.n = pbar.total
            pbar.refresh()
            print(info)
            break

print("Total Episode Reward:", episode_reward)

# === CREATE HEATMAP NEXT TO ACMI FILE ===
heatmap_path = os.path.splitext(acmi_filepath)[0] + "_heatmap.png"
plot_path = os.path.splitext(acmi_filepath)[0] + "_flight_path.html"
heatmap = FlightHeatmap(acmi_filepath)
heatmap.plot_3d(save_path=heatmap_path, show=False)
#heatmap.plotly_3d(save_path=plot_path, show=False)
print(f"Heatmap saved to {heatmap_path}")