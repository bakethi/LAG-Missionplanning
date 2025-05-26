import os
import sys
import numpy as np

# Add the project root to the Python path
# Assuming this script is in LAG-Missionplanning/scripts/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import necessary classes
from envs.JSBSim.envs import SingleCombatEnv
from envs.JSBSim.utils.utils import parse_config

def run_test_logic():
    print("--- Starting Test Logic Script ---")

    # 1. Load Configuration
    # Ensure this path matches your OpStory01.yaml location relative to envs/JSBSim/configs/
    config_name = "OpStories/OpStory01"
    
    # parse_config expects the base path to be envs/JSBSim/configs/
    # So, we pass the relative path from that directory
    config = parse_config(config_name)

    print(f"\nConfiguration loaded from: envs/JSBSim/configs/{config_name}.yaml")
    print(f"Airbase Initial HP: {config.airbase_initial_hp}")
    print(f"Airbase Position: {config.airbase_position}")
    print(f"Missile Damage: {config.missile_damage}")
    print(f"Airbase Destruction Reward: {config.airbase_destruction_reward}")

    # 2. Instantiate the Environment
    # The SingleCombatEnv will use the loaded config to set up the task and airbase
    env = SingleCombatEnv(config_name)

    # 3. Print Observation and Action Space
    print("\n--- Environment Spaces ---")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Number of Agents: {env.num_agents}")

    # 4. Run a Test Episode
    print("\n--- Running Test Episode ---")
    obs = env.reset()
    print(f"Episode Reset. Initial Airbase HP: {env.airbase['hp']}") # Direct access to env.airbase
    print(f"Initial Obs (Agent 0): {obs[0]}") # Assuming single agent, obs is packed as array

    total_reward = 0
    max_test_steps = 1000 # Adjust this to run for more or fewer steps

    for step_num in range(max_test_steps):
        # Generate a random action
        # This simulates a randomly behaving agent.
        # For a specific attack test, you might replace this with a hardcoded action
        # that moves towards the airbase and eventually fires a missile (if enabled).
        action = np.array([env.action_space.sample()]) # Sample a random action, wrap in array for single agent
        print(action)
        # If you've modified OpStory01_task.py to have a 5th action for firing:
        # Example to force fire a missile after a certain number of steps:
        if step_num == 100:
            print("\n--- FORCING MISSILE FIRE AT STEP 100 (if action space supports it) ---")
            # Assuming action_space.nvec for aileron, elevator, rudder, throttle, fire_weapon
            # And 'fire_weapon' is the last discrete action with 2 options (0=no_fire, 1=fire)
            action = np.array([[20, 20, 20, 15, 1]]) # Example of a fire action
        else:
            action = np.array([[20, 20, 20, 15, 0]]) # Example of no-fire action


        packed_obs, packed_rewards, packed_dones, info = env.step(action)
        
        # Unpack rewards, dones for single agent
        reward = packed_rewards[0][0]
        done = packed_dones[0][0]
        obs = packed_obs

        total_reward += reward

        # Print relevant information
        print(f"Step {step_num + 1}: Reward={reward:.2f}, Total_Reward={total_reward:.2f}, "
              f"Airbase HP={info.get('airbase_hp', 'N/A')}, Airbase Destroyed={info.get('airbase_destroyed', 'N/A')}, "
              f"Done={done}")
        # print(f"  Obs (Agent 0, first 5 elements): {obs[0][:5]}") # Print part of observation

        if done:
            print(f"\nEpisode finished at step {step_num + 1}. Final info: {info}")
            break

    env.close()
    print("\n--- Test Logic Script Finished ---")

if __name__ == "__main__":
    run_test_logic()