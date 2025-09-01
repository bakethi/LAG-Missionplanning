import numpy as np
import torch
import random
import sys, os
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.JSBSim.envs import MultiWaypointEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging

# Import the heatmap class
from renders.heatmap import FlightHeatmap

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

class FlightDataCollector:
    """Advanced flight data collection and visualization for multiwaypoint missions"""
    
    def __init__(self, max_steps=3000):
        self.max_steps = max_steps
        self.reset_data()
    
    def reset_data(self):
        """Initialize/reset all data arrays"""
        self.time_data = []
        self.position_data = []
        self.velocity_data = []
        self.attitude_data = []
        self.airspeed_data = []
        self.altitude_data = []
        self.gforce_data = []
        self.mass_data = []
        self.waypoint_data = []
        
        # Reward-relevant metrics
        self.distance_to_waypoint = []
        self.yaw_difference = []
        self.alignment_score = []
        
        self.step_count = 0
    
    def collect_step_data(self, env):
        """Collect flight data for current simulation step"""
        # Get the first (and only) aircraft from the agents dictionary
        agent_id = list(env.agents.keys())[0]
        aircraft = env.agents[agent_id]
        
        # Get current simulation time
        current_time = aircraft.get_sim_time()
        self.time_data.append(current_time)
        
        # Extract position data
        pos = aircraft.get_position()
        self.position_data.append([pos[0], pos[1], pos[2]])
        
        # Extract velocity data
        vel = aircraft.get_velocity()
        self.velocity_data.append([vel[0], vel[1], vel[2]])
        
        # Extract attitude data (roll, pitch, yaw)
        rpy = aircraft.get_rpy()
        self.attitude_data.append([rpy[0], rpy[1], rpy[2]])
        
        # Extract additional flight parameters
        try:
            # Get airspeed
            airspeed = aircraft.get_property_value(c.velocities_vc_mps)
            
            # Get altitude
            altitude = aircraft.get_property_value(c.position_h_sl_m)
            
            # Get G-force (load factor)
            gforce = aircraft.get_property_value(c.accelerations_n_pilot_z_norm)
            
            # Get mass
            mass = aircraft.get_property_value(c.inertia_weight_lbs) * 0.453592  # lbs to kg
            
        except:
            # Use default values if properties are not available
            airspeed = np.sqrt(sum([v**2 for v in vel]))
            altitude = pos[2]
            gforce = 1.0
            mass = 10000.0  # Default mass in kg
        
        # Append the collected data
        self.airspeed_data.append(airspeed)
        self.altitude_data.append(altitude)
        self.gforce_data.append(gforce)
        self.mass_data.append(mass)
        
        # Record current waypoint information
        waypoint_info = {
            'step': self.step_count,
            'time': current_time,
            'active_waypoint': env.active_waypoint,
            'task_stage': env.task_stage,
            'total_waypoints': len(env.waypoint_sequence)
        }
        self.waypoint_data.append(waypoint_info)
        
        # Collect reward-relevant metrics
        self.collect_reward_metrics(env, agent_id)
        
        self.step_count += 1
    
    def collect_reward_metrics(self, env, agent_id):
        """Collect metrics that are relevant for reward calculation"""
        try:
            # Distance to waypoint
            distance = env.compute_distance_to_waypoint(agent_id)
            self.distance_to_waypoint.append(distance)
            
            # Yaw difference (alignment to waypoint)
            yaw_diff = env.get_alignment_to_waypoint(agent_id)
            self.yaw_difference.append(yaw_diff)
            
            # Alignment score (Gaussian reward component)
            yaw_sigma = 0.25  # from reward function
            alignment_score = np.exp(-(yaw_diff / yaw_sigma) ** 2)
            self.alignment_score.append(alignment_score)
            
        except Exception as e:
            # Use default values if metrics can't be computed
            self.distance_to_waypoint.append(float('inf'))
            self.yaw_difference.append(0.0)
            self.alignment_score.append(0.0)
            print(f"Warning: Could not collect reward metrics - {e}")
    
    def convert_to_numpy(self):
        """Convert collected data to numpy arrays for analysis"""
        self.time_data = np.array(self.time_data)
        self.position_data = np.array(self.position_data)
        self.velocity_data = np.array(self.velocity_data)
        self.attitude_data = np.array(self.attitude_data)
        self.airspeed_data = np.array(self.airspeed_data)
        self.altitude_data = np.array(self.altitude_data)
        self.gforce_data = np.array(self.gforce_data)
        self.mass_data = np.array(self.mass_data)
        
        # Convert reward metrics to numpy arrays
        self.distance_to_waypoint = np.array(self.distance_to_waypoint)
        self.yaw_difference = np.array(self.yaw_difference)
        self.alignment_score = np.array(self.alignment_score)
    
    def save_matlab_data(self, save_path):
        """Save collected data in MATLAB format"""
        import scipy.io as sio
        
        self.convert_to_numpy()
        
        matlab_data = {
            'time_data': self.time_data,
            'position_data': self.position_data,
            'velocity_data': self.velocity_data,
            'attitude_data': self.attitude_data,
            'airspeed_data': self.airspeed_data,
            'altitude_data': self.altitude_data,
            'gforce_data': self.gforce_data,
            'mass_data': self.mass_data,
            'waypoint_data': self.waypoint_data,
            
            # Reward-relevant metrics
            'distance_to_waypoint': self.distance_to_waypoint,
            'yaw_difference': self.yaw_difference,
            'alignment_score': self.alignment_score
        }
        
        sio.savemat(save_path, matlab_data)
        print(f"MATLAB data saved to {save_path}")
    
    def create_advanced_plots(self, save_path=None):
        """Create comprehensive flight visualization plots"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        self.convert_to_numpy()

        # Extract data
        north = self.position_data[:, 0]
        east = self.position_data[:, 1]
        up = self.position_data[:, 2]

        vel_north = self.velocity_data[:, 0]
        vel_east = self.velocity_data[:, 1]
        vel_up = self.velocity_data[:, 2]

        roll = self.attitude_data[:, 0]
        pitch = self.attitude_data[:, 1]
        yaw = self.attitude_data[:, 2]

        # Calculate derived quantities
        ground_speed = np.sqrt(vel_north**2 + vel_east**2)
        total_speed = np.sqrt(vel_north**2 + vel_east**2 + vel_up**2)
        climb_rate = vel_up

        # Create figure with subplots - now 2x3 grid
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('Advanced Flight Analysis Dashboard - Multiwaypoint Mission with Reward Metrics', fontsize=16, fontweight='bold')

        # Get waypoint times for vertical lines
        waypoint_times = {}
        for wp_info in self.waypoint_data:
            stage = wp_info['task_stage']
            if stage > 0 and stage not in waypoint_times:  # Only actual waypoints
                waypoint_times[stage] = wp_info['time']

        # 1. 3D Flight Path with Waypoints (z-axis starts at 0)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(north, east, up, 'b-', linewidth=2, label='Flight Path')
        ax1.scatter(north[0], east[0], up[0], c='green', s=100, marker='o', label='Start')
        ax1.scatter(north[-1], east[-1], up[-1], c='red', s=100, marker='o', label='End')

        # Add waypoint markers (exclude starting point which is stage 0)
        waypoint_stages = {}
        for wp_info in self.waypoint_data:
            stage = wp_info['task_stage']
            if stage > 0 and stage not in waypoint_stages:  # Only actual waypoints
                waypoint_stages[stage] = wp_info['step']

        for stage, step in waypoint_stages.items():
            if step < len(north):
                ax1.scatter(north[step], east[step], up[step], c='orange', s=60, marker='^', alpha=0.8)

        ax1.set_xlabel('North (m)')
        ax1.set_ylabel('East (m)')
        ax1.set_zlabel('Up (m)')
        ax1.set_zlim(0, np.max(up) * 1.05)
        ax1.set_title('3D Flight Path with Waypoints')
        ax1.legend()
        ax1.grid(True)

        # 2. Distance to Waypoint over Time
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(self.time_data, self.distance_to_waypoint / 1000, 'r-', linewidth=2, label='Distance to Waypoint')
        ax2.axhline(y=0.75, color='g', linestyle='--', alpha=0.7, label='Success Radius (750m)')

        # Add waypoint vertical lines
        for stage, wp_time in waypoint_times.items():
            ax2.axvline(x=wp_time, color='orange', linestyle='--', alpha=0.7, linewidth=1)
            ax2.text(wp_time, ax2.get_ylim()[1] * 0.9, f'WP{stage}', rotation=90, ha='right', va='top')

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Distance (km)')
        ax2.set_title('Distance to Active Waypoint')
        ax2.legend()
        ax2.grid(True)

        # 3. Yaw Difference and Alignment Score
        ax3 = fig.add_subplot(2, 3, 3)
        ax3_twin = ax3.twinx()
        ax3.plot(self.time_data, self.yaw_difference * 180/np.pi, 'b-', linewidth=2, label='Yaw Difference')
        ax3_twin.plot(self.time_data, self.alignment_score, 'g-', linewidth=2, label='Alignment Score')

        # Add waypoint vertical lines
        for stage, wp_time in waypoint_times.items():
            ax3.axvline(x=wp_time, color='orange', linestyle='--', alpha=0.7, linewidth=1)
            ax3.text(wp_time, ax3.get_ylim()[1] * 0.9, f'WP{stage}', rotation=90, ha='right', va='top')

        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Yaw Difference (degrees)', color='b')
        ax3_twin.set_ylabel('Alignment Score', color='g')
        ax3.set_title('Waypoint Alignment')
        ax3.axhline(y=0, color='b', linestyle='--', alpha=0.3)
        ax3.grid(True)

        # 4. Attitude Details (Roll, Pitch, Yaw)
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(self.time_data, roll * 180/np.pi, 'r-', linewidth=2, label='Roll')
        ax4.plot(self.time_data, pitch * 180/np.pi, 'g-', linewidth=2, label='Pitch')
        ax4.plot(self.time_data, yaw * 180/np.pi, 'b-', linewidth=2, label='Yaw')

        # Add waypoint vertical lines
        for stage, wp_time in waypoint_times.items():
            ax4.axvline(x=wp_time, color='orange', linestyle='--', alpha=0.7, linewidth=1)
            ax4.text(wp_time, ax4.get_ylim()[1] * 0.9, f'WP{stage}', rotation=90, ha='right', va='top')

        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angle (degrees)')
        ax4.set_title('Aircraft Attitude')
        ax4.legend()
        ax4.grid(True)

        # 5. Ground Track (Top View) with Waypoints
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(east, north, 'b-', linewidth=2, label='Flight Path')
        ax5.scatter(east[0], north[0], c='green', s=100, marker='o', label='Start')
        ax5.scatter(east[-1], north[-1], c='red', s=100, marker='o', label='End')

        # Add waypoint markers to ground track
        waypoint_counter = 1
        for stage in sorted(waypoint_stages.keys()):
            step = waypoint_stages[stage]
            if step < len(east):
                ax5.scatter(east[step], north[step], c='orange', s=60, marker='^', alpha=0.8)
                ax5.annotate(f'WP{waypoint_counter}', (east[step], north[step]), xytext=(5, 5), textcoords='offset points')
                waypoint_counter += 1

        ax5.set_xlabel('East (m)')
        ax5.set_ylabel('North (m)')
        ax5.set_title('Ground Track (Top View)')
        ax5.legend()
        ax5.grid(True)
        ax5.axis('equal')

        # 6. Velocity and Performance
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(self.time_data, ground_speed, 'b-', linewidth=2, label='Ground Speed')
        ax6.plot(self.time_data, self.airspeed_data, 'r-', linewidth=2, label='Airspeed')
        ax6.plot(self.time_data, climb_rate, 'g-', linewidth=2, label='Climb Rate')

        # Add waypoint vertical lines
        for stage, wp_time in waypoint_times.items():
            ax6.axvline(x=wp_time, color='orange', linestyle='--', alpha=0.7, linewidth=1)
            ax6.text(wp_time, ax6.get_ylim()[1] * 0.9, f'WP{stage}', rotation=90, ha='right', va='top')

        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Speed (m/s)')
        ax6.set_title('Velocity Components')
        ax6.legend()
        ax6.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Advanced plots saved to {save_path}")

        plt.show()
    
    def print_mission_summary(self):
        """Print comprehensive mission summary statistics"""
        self.convert_to_numpy()
        
        # Extract data
        north = self.position_data[:, 0]
        east = self.position_data[:, 1]
        up = self.position_data[:, 2]
        
        vel_north = self.velocity_data[:, 0]
        vel_east = self.velocity_data[:, 1]
        vel_up = self.velocity_data[:, 2]
        
        ground_speed = np.sqrt(vel_north**2 + vel_east**2)
        
        print("\n" + "="*50)
        print("MULTIWAYPOINT MISSION SUMMARY STATISTICS")
        print("="*50)
        print(f"Mission Duration: {self.time_data[-1]:.1f} seconds")
        
        # Get total waypoints from environment
        if self.waypoint_data:
            total_waypoints_from_env = self.waypoint_data[0]['total_waypoints']
            print(f"Total Waypoints: {total_waypoints_from_env}")
        else:
            print("Total Waypoints: Unknown")
        
        print(f"Total Distance: {np.sum(np.sqrt(np.diff(north)**2 + np.diff(east)**2 + np.diff(up)**2)):.1f} meters")
        print(f"Max Altitude: {np.max(self.altitude_data):.1f} meters")
        print(f"Min Altitude: {np.min(self.altitude_data):.1f} meters")
        print(f"Max Airspeed: {np.max(self.airspeed_data):.1f} m/s")
        print(f"Max Ground Speed: {np.max(ground_speed):.1f} m/s")
        print(f"Max Climb Rate: {np.max(vel_up):.1f} m/s")
        print(f"Max G-Force: {np.max(self.gforce_data):.2f}")
        print(f"Average Mass: {np.mean(self.mass_data):.1f} kg")
        print(f"Average Airspeed: {np.mean(self.airspeed_data):.1f} m/s")
        print(f"Average Ground Speed: {np.mean(ground_speed):.1f} m/s")
        print("="*50)
        
        # Waypoint-specific statistics
        print("\nWAYPOINT PROGRESSION:")
        waypoint_times = {}
        for wp_info in self.waypoint_data:
            stage = wp_info['task_stage']
            if stage not in waypoint_times:
                waypoint_times[stage] = wp_info['time']
        
        # Only show actual waypoints (exclude starting point which is stage 0)
        actual_stages = [stage for stage in sorted(waypoint_times.keys()) if stage > 0]
        
        for i, stage in enumerate(actual_stages):
            print(f"  Waypoint {i+1}: Reached at {waypoint_times[stage]:.1f}s")
        
        # Note about final waypoint if not captured in task_stage progression
        if len(actual_stages) < 3:  # We know there should be 3 waypoints total
            print(f"  Note: Final waypoint reached at end of mission ({self.time_data[-1]:.1f}s)")
        
        print("="*50)
        
        # Reward-relevant metrics summary
        print("\nREWARD METRICS SUMMARY:")
        print(f"Average Distance to Waypoint: {np.mean(self.distance_to_waypoint):.1f} m")
        print(f"Min Distance to Waypoint: {np.min(self.distance_to_waypoint):.1f} m")
        print(f"Max Distance to Waypoint: {np.max(self.distance_to_waypoint):.1f} m")
        print(f"Average Yaw Difference: {np.mean(np.abs(self.yaw_difference)) * 180/np.pi:.1f} degrees")
        print(f"Max Yaw Difference: {np.max(np.abs(self.yaw_difference)) * 180/np.pi:.1f} degrees")
        print(f"Average Alignment Score: {np.mean(self.alignment_score):.3f}")
        
        # Time spent in different alignment conditions
        well_aligned = np.sum(np.abs(self.yaw_difference) < np.radians(15))
        poorly_aligned = np.sum(np.abs(self.yaw_difference) > np.radians(45))
        total_steps = len(self.yaw_difference)
        
        print(f"\nAlignment Analysis:")
        print(f"  Well aligned (<15°): {well_aligned/total_steps*100:.1f}% of mission")
        print(f"  Poorly aligned (>45°): {poorly_aligned/total_steps*100:.1f}% of mission")
        
        print("="*50)

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
env_name = env.__class__.__name__
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
acmi_dir = os.path.join(".", env_name, timestamp)
os.makedirs(acmi_dir, exist_ok=True)
acmi_filepath = os.path.join(acmi_dir, f"{experiment_name}.txt.acmi")

# === FLIGHT DATA COLLECTOR SETUP ===
flight_data = FlightDataCollector(max_steps=env.max_steps)

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
print("Starting multiwaypoint mission with advanced data collection...")
with tqdm(total=env.max_steps, desc="Running Simulation", unit="step", ncols=100) as pbar:
    while True:
        # Collect flight data for this step
        flight_data.collect_step_data(env)
        
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

# === ADVANCED DATA ANALYSIS AND VISUALIZATION ===
print("\nProcessing flight data and creating visualizations...")

# Generate comprehensive plots
plots_path = os.path.join(acmi_dir, f"{experiment_name}_advanced_plots.png")
flight_data.create_advanced_plots(save_path=plots_path)

# Save MATLAB data
matlab_path = os.path.join(acmi_dir, f"{experiment_name}_flight_data.mat")
flight_data.save_matlab_data(matlab_path)

# Print mission summary
flight_data.print_mission_summary()

# === CREATE ORIGINAL HEATMAP ===
heatmap_path = os.path.splitext(acmi_filepath)[0] + "_heatmap.png"
heatmap = FlightHeatmap(acmi_filepath)
heatmap.plot_3d(save_path=heatmap_path, show=False)
print(f"Original heatmap saved to {heatmap_path}")

print(f"\nAll outputs saved to: {acmi_dir}")
print("Files generated:")
print(f"  - ACMI file: {acmi_filepath}")
print(f"  - Advanced plots: {plots_path}")
print(f"  - MATLAB data: {matlab_path}")
print(f"  - Heatmap: {heatmap_path}")
