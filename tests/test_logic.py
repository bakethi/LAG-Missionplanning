import os
import sys
import numpy as np
import logging

# Define the log folder path
LOG_FOLDER = "logs" # You can change this to any desired folder name

# Create the log folder if it doesn't exist
os.makedirs(LOG_FOLDER, exist_ok=True)

# Define the full path for the log file
log_file_path = os.path.join(LOG_FOLDER, "test_logic_output.log")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, # Set desired logging level (e.g., logging.DEBUG for more verbosity)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), # Output to console
        logging.FileHandler(log_file_path) # Output to the specified log file in the folder
    ]
)

# Get a logger instance for your script
logger = logging.getLogger(__name__)

# Add the project root to the Python path
# Assuming this script is in LAG-Missionplanning/scripts/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import necessary classes
from envs.JSBSim.envs import SingleCombatEnv
from envs.JSBSim.utils.utils import parse_config

def run_test_logic():
    logger.info("--- Starting Test Logic Script ---")

    # 1. Load Configuration
    # Ensure this path matches your OpStory01.yaml location relative to envs/JSBSim/configs/
    config_name = "OpStories/OpStory01"

    # parse_config expects the base path to be envs/JSBSim/configs/
    # So, we pass the relative path from that directory
    config = parse_config(config_name)

    logger.info(f"\nConfiguration loaded from: envs/JSBSim/configs/{config_name}.yaml")
    logger.info(f"Airbase Initial HP: {config.airbase_initial_hp}")
    logger.info(f"Airbase Position: {config.airbase_position}")
    logger.info(f"Missile Damage: {config.missile_damage}")
    logger.info(f"Airbase Destruction Reward: {config.airbase_destruction_reward}")

    # 2. Instantiate the Environment
    # The SingleCombatEnv will use the loaded config to set up the task and airbase
    env = SingleCombatEnv(config_name)

    # 3. Print Observation and Action Space
    logger.info("\n--- Environment Spaces ---")
    logger.info(f"Observation Space: {env.observation_space}")
    logger.info(f"Action Space: {env.action_space}")
    logger.info(f"Number of Agents: {env.num_agents}")

    # 4. Run a Test Episode
    logger.info("\n--- Running Test Episode ---")
    obs = env.reset()
    logger.info(f"Episode Reset. Initial Airbase HP: {env.airbase['hp']}")
    logger.info(f"Initial Obs (Agent 0): {obs[0]}")

    total_reward = 0
    max_test_steps = 1000 # Adjust this to run for more or fewer steps

    for step_num in range(max_test_steps):
        # Generate a random action
        # This simulates a randomly behaving agent.
        # For a specific attack test, you might replace this with a hardcoded action
        # that moves towards the airbase and eventually fires a missile (if enabled).
        action = np.array([env.action_space.sample()])
        # If you've modified OpStory01_task.py to have a 5th action for firing:
        # Example to force fire a missile after a certain number of steps:
        if step_num == 100:
            logger.info("\n--- FORCING MISSILE FIRE AT STEP 100 (if action space supports it) ---")
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
        logger.info(f"Step {step_num + 1}: Reward={reward:.2f}, Total_Reward={total_reward:.2f}, "
              f"Airbase HP={info.get('airbase_hp', 'N/A')}, Airbase Destroyed={info.get('airbase_destroyed', 'N/A')}, "
              f"Done={done}")
        # logger.info(f"  Obs (Agent 0, first 5 elements): {obs[0][:5]}") # Print part of observation

        if done:
            logger.info(f"\nEpisode finished at step {step_num + 1}. Final info: {info}")
            break

    env.close()
    logger.info("\n--- Test Logic Script Finished ---")

if __name__ == "__main__":
    run_test_logic()