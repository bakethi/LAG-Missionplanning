import os, sys, math, json, argparse, random
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.JSBSim.envs import MultiWaypointEnv
from algorithms.ppo.ppo_actor import PPOActor

# -------------------- Utils --------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

class ArgsPolicy:
    def __init__(self):
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

def parse_args():
    p = argparse.ArgumentParser("Clean MultiWaypoint Evaluation for Thesis")
    p.add_argument('--model-dir', type=str, required=True, help='Directory containing actor_latest.pt or actor_<index>.pt')
    p.add_argument('--policy-index', type=str, default='latest')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--target-waypoints', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--render', action='store_true', help='Save last episode ACMI + heatmap (if heatmap module available)')
    p.add_argument('--out-dir', type=str, default='evaluation_outputs')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--success-radius', type=float, default=750.0, help='Waypoint success radius (m) for optimal distance adjustment')
    return p.parse_args()

def build_optimal_distances(env, target_waypoints, success_radius=0.0):
    """Build optimal path distances considering success radius."""
    ordered_uids = [env.waypoint_sequence[i] for i in range(len(env.waypoint_sequence))]
    tgt = min(target_waypoints, len(ordered_uids))
    if tgt <= 1:
        return 0.0, 0.0  # path (wp->wp), chain (start+path)
    
    # Hop distances (center-to-center minus success radius per destination)
    hop_dists = []
    for i in range(tgt - 1):
        a = env.waypoints[ordered_uids[i]].get_geodetic()  # lon,lat,alt
        b = env.waypoints[ordered_uids[i+1]].get_geodetic()
        center_dist = haversine_m(a[1], a[0], b[1], b[0])
        effective_dist = max(0.0, center_dist - success_radius)
        hop_dists.append(effective_dist)
    
    path_wp_wp = sum(hop_dists)
    
    # Add start->first wp (also adjusted by radius)
    start_geo = list(env.agents.values())[0].get_geodetic().copy()
    first_wp = env.waypoints[ordered_uids[0]].get_geodetic()
    start_center = haversine_m(start_geo[1], start_geo[0], first_wp[1], first_wp[0])
    start_effective = max(0.0, start_center - success_radius)
    chain = start_effective + path_wp_wp
    
    return path_wp_wp, chain

# -------------------- Core Evaluation --------------------

def evaluate():
    cli = parse_args()
    random.seed(cli.seed); np.random.seed(cli.seed); torch.manual_seed(cli.seed)

    env = MultiWaypointEnv("eval/multiwaypoint")
    env.seed(cli.seed)

    # Time scaling: 252s per waypoint
    step_seconds = env.agent_interaction_steps / env.sim_freq
    env.max_steps = int((252 * cli.target_waypoints) / step_seconds)

    # Policy
    policy_args = ArgsPolicy()
    policy = PPOActor(policy_args, env.observation_space, env.action_space, device=torch.device('cpu'))
    model_path = os.path.join(cli.model_dir, f"actor_{cli.policy_index}.pt")
    policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy.eval()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = os.path.join(cli.out_dir, f"thesis_eval_{ts}")
    os.makedirs(out_root, exist_ok=True)

    # Episode metrics
    ep_rewards = []
    ep_completion_times = []  # seconds for successful episodes
    ep_distances = []         # total geodesic distance
    ep_distance_ratios = []   # distance / optimal
    ep_waypoints = []         # waypoints reached
    ep_success = []           # mission success
    ep_steps = []            # steps taken
    
    last_acmi = None

    for ep in tqdm(range(cli.episodes), desc='Evaluating', ncols=90):
        obs = env.reset()
        opt_path, opt_chain = build_optimal_distances(env, cli.target_waypoints, cli.success_radius)
        
        rnn_states = np.zeros((1,1,policy_args.recurrent_hidden_size), dtype=np.float32)
        masks = np.ones((1,1), dtype=np.float32)
        total_reward = 0.0
        geods = []
        cum_dist = [0.0]
        prev_stage = env.task_stage

        if cli.render and ep == cli.episodes - 1:
            env_name = env.__class__.__name__
            acmi_dir = os.path.join(out_root, env_name)
            os.makedirs(acmi_dir, exist_ok=True)
            last_acmi = os.path.join(acmi_dir, f"thesis_ep{ep}.acmi")
            env.render(mode='txt', filepath=last_acmi)

        for step in range(env.max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32)
            masks_t = torch.tensor(masks, dtype=torch.float32)
            rnn_t = torch.tensor(rnn_states, dtype=torch.float32)
            with torch.no_grad():
                action, _, rnn_t = policy(obs_t, rnn_t, masks_t, deterministic=True)
            action_np = action.detach().cpu().numpy()
            rnn_states = rnn_t.detach().cpu().numpy()
            obs, reward, done, info = env.step(action_np)
            total_reward += float(np.array(reward).squeeze())
            
            geo = list(env.agents.values())[0].get_geodetic().copy()
            geods.append(geo)
            if len(geods) > 1:
                a = geods[-2]; b = geods[-1]
                seg = haversine_m(a[1], a[0], b[1], b[0])
                cum_dist.append(cum_dist[-1] + seg)
            
            if cli.render and ep == cli.episodes - 1:
                env.render(mode='txt', filepath=last_acmi)
            
            # Check mission progress
            if env.task_stage > prev_stage:
                prev_stage = env.task_stage
                if env.task_stage >= cli.target_waypoints:
                    break
            if done.all():
                break

        # Episode metrics
        steps_taken = step + 1
        total_distance = cum_dist[-1]
        visited = min(env.task_stage, cli.target_waypoints)
        success = visited >= cli.target_waypoints
        completion_time = steps_taken * step_seconds if success else None
        ratio = (total_distance / opt_path) if opt_path > 0 else np.nan
        
        ep_rewards.append(total_reward)
        ep_completion_times.append(completion_time)
        ep_distances.append(total_distance)
        ep_distance_ratios.append(ratio)
        ep_waypoints.append(visited)
        ep_success.append(success)
        ep_steps.append(steps_taken)
        
        if cli.debug:
            print(f"[EP {ep}] reward={total_reward:.1f} dist={total_distance:.1f} ratio={ratio:.2f} visited={visited} success={success}")

    # Summary statistics
    successful_times = [t for t in ep_completion_times if t is not None]
    
    summary = {
        'episodes': cli.episodes,
        'target_waypoints': cli.target_waypoints,
        'success_rate': float(np.mean(ep_success)),
        'avg_reward': float(np.mean(ep_rewards)),
        'avg_distance_m': float(np.mean(ep_distances)),
        'avg_distance_ratio': float(np.nanmean(ep_distance_ratios)),
        'avg_completion_time_s': float(np.mean(successful_times)) if successful_times else None,
        'avg_steps': float(np.mean(ep_steps)),
        'success_radius_m': cli.success_radius,
    }
    
    with open(os.path.join(out_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_root, 'summary.txt'), 'w') as f:
        for k,v in summary.items():
            f.write(f"{k}: {v}\n")

    # =============== Thesis Plots ===============
    def savefig(fig, name):
        fig.savefig(os.path.join(out_root, name), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 1. Mission Success Rate vs Episode (running average)
    if len(ep_success) > 1:
        window = max(1, len(ep_success) // 10)
        running_success = []
        for i in range(len(ep_success)):
            start_idx = max(0, i - window + 1)
            running_success.append(np.mean(ep_success[start_idx:i+1]))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(running_success)), running_success, 'b-', linewidth=2)
        ax.axhline(1.0, color='g', linestyle='--', alpha=0.7, label='Perfect Success')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate (Running Average)')
        ax.set_title('Mission Success Rate Over Episodes')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1.1)
        savefig(fig, 'mission_success_rate.png')

    # 2. Path Efficiency Distribution
    clean_ratios = [r for r in ep_distance_ratios if not np.isnan(r)]
    if clean_ratios:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(clean_ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Optimal (1.0)')
        ax1.axvline(np.mean(clean_ratios), color='orange', linestyle='-', linewidth=2, label=f'Mean ({np.mean(clean_ratios):.2f})')
        ax1.set_xlabel('Path Efficiency Ratio (Actual/Optimal)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Path Efficiency Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Time series
        ax2.plot(range(len(clean_ratios)), clean_ratios, 'o-', alpha=0.7, markersize=4)
        ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Path Efficiency Ratio')
        ax2.set_title('Path Efficiency Over Episodes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        savefig(fig, 'path_efficiency_analysis.png')

    # 3. Reward vs Distance Correlation
    if ep_rewards and ep_distances:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(ep_distances, ep_rewards, c=ep_success, cmap='RdYlGn', 
                           s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Total Distance Traveled (m)')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Reward vs Distance Relationship')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Mission Success (0=Fail, 1=Success)')
        
        # Add correlation coefficient
        valid_indices = [i for i in range(len(ep_rewards)) if not np.isnan(ep_distances[i])]
        if len(valid_indices) > 1:
            corr = np.corrcoef([ep_distances[i] for i in valid_indices], 
                             [ep_rewards[i] for i in valid_indices])[0,1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        savefig(fig, 'reward_distance_correlation.png')

    # 4. Completion Time Analysis (successful episodes only)
    if successful_times:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(successful_times, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax1.axvline(np.mean(successful_times), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(successful_times):.1f}s')
        ax1.set_xlabel('Completion Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Mission Completion Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CDF
        sorted_times = np.sort(successful_times)
        y_vals = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax2.plot(sorted_times, y_vals, 'b-', linewidth=2)
        ax2.axvline(np.median(successful_times), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(successful_times):.1f}s')
        ax2.set_xlabel('Completion Time (seconds)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Completion Time CDF')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        savefig(fig, 'completion_time_analysis.png')

    # 5. Performance Summary Box Plot
    metrics_data = {
        'Rewards': ep_rewards,
        'Path Ratios': clean_ratios,
        'Completion Times': [t/60 for t in successful_times],  # Convert to minutes
        'Waypoints Reached': ep_waypoints
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, data) in enumerate(metrics_data.items()):
        if data:
            axes[i].boxplot(data, patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
            axes[i].set_title(f'{metric} Distribution')
            axes[i].set_ylabel(metric.replace('Times', 'Times (min)'))
            axes[i].grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = np.mean(data)
            axes[i].axhline(mean_val, color='red', linestyle='--', alpha=0.7,
                          label=f'Mean: {mean_val:.2f}')
            axes[i].legend()
    
    plt.tight_layout()
    savefig(fig, 'performance_summary_boxplot.png')

    print('=== THESIS EVALUATION SUMMARY ===')
    for k,v in summary.items():
        print(f"{k}: {v}")
    if last_acmi:
        print(f"Last episode ACMI: {last_acmi}")
    print(f"Artifacts written to: {out_root}")

if __name__ == '__main__':
    evaluate()
