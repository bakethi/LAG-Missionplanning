#!/bin/sh

# Script to train the OpStory01 airbase attack scenario

# --- Configuration Variables ---
# The --env-name argument for train_jsbsim.py (e.g., "SingleCombat" if using SingleCombatEnv)
env_type="SingleCombat"

# The --scenario-name argument for train_jsbsim.py
# This points to your YAML config file, relative to envs/JSBSim/configs/
scenario_config_path="OpStories/OpStory01" # MODIFY THIS AS NEEDED

# RL algorithm to use (can be dummy for testing)
rl_algorithm="ppo"

# A descriptive name for this experiment run
experiment_tag="OpStory01_LogicTest_Run" # Changed for testing

# Random seed for reproducibility
random_seed=1

# Training resources
num_training_threads=1    # Keep at 1 for simplicity during testing
num_rollout_threads=1     # Set to 1 for easier debugging and sequential execution
use_cuda_flag=""          # Disable GPU for simpler CPU-based testing. Use "--cuda" to enable.

# Training duration and PPO hyperparameters
# REDUCED TOTAL STEPS FOR TESTING
total_env_steps=5000      # <<<<<<<<<<<<<<<<<<<<<< IMPORTANT: Drastically reduced for testing
ppo_epochs=1              # Minimal PPO epochs
num_mini_batches=1        # Minimal mini-batches
data_chunk_len=1          # Minimal data chunk length
learning_rate=0           # Set to 0 or very small if you don't want any learning updates
discount_factor_gamma=0.0 # Not critical for logic testing, but can be set
ppo_clip_param=0.0        # Not critical for logic testing, can be set to 0
max_grad_norm_val=0.0     # Not critical for logic testing, can be set to 0
entropy_coef_val=0.0      # Not critical for logic testing, can be set to 0

# Logging and Saving
log_update_interval=1     # Log every update to see frequent output
save_model_interval=10000 # Effectively disable model saving by setting a very high interval
use_wandb_logging=""      # <<<<<<<<<<<<<<<<<<<<<< IMPORTANT: Disable WandB for clean console output during testing
wandb_project_id="your_wandb_project_name"
user_identifier="your_username"

# Set render_mode to 'txt' to see console output
render_mode="txt" # This is already the default for train_jsbsim.py, but explicit is good

# --- End Configuration Variables ---

# Inform the user about the settings
echo "======================================================================================"
echo "Starting Logic Test for Operational Story 01 (Airbase Attack)"
echo "--------------------------------------------------------------------------------------"
echo "Environment Type         : ${env_type}"
echo "Scenario Config Path     : ${scenario_config_path}"
echo "RL Algorithm             : ${rl_algorithm}"
echo "Experiment Tag           : ${experiment_tag}"
echo "Random Seed              : ${random_seed}"
echo "Rollout Threads          : ${num_rollout_threads}"
echo "Total Environment Steps  : ${total_env_steps}"
echo "Using CUDA               : ${use_cuda_flag:-"No (CPU)"}"
echo "W&B Logging              : ${use_wandb_logging:-"No"}"
echo "Render Mode              : ${render_mode}" # Added explicit render mode
echo "======================================================================================"

# Construct the path to the training script
current_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
training_script_path="${current_script_dir}/tests/test_logic.py"

# Check if the training script exists
if [ ! -f "${training_script_path}" ]; then
    echo "ERROR: Training script not found at ${training_script_path}"
    echo "Please ensure this script is in the 'LAG-Missionplanning/scripts/' directory."
    exit 1
fi

# Execute the training script
python ${training_script_path} \
    --env-name "${env_type}" \
    --algorithm-name "${rl_algorithm}" \
    --scenario-name "${scenario_config_path}" \
    --experiment-name "${experiment_tag}" \
    --seed ${random_seed} \
    --n-training-threads ${num_training_threads} \
    --n-rollout-threads ${num_rollout_threads} \
    ${use_cuda_flag} \
    --log-interval ${log_update_interval} \
    --save-interval ${save_model_interval} \
    --num-env-steps ${total_env_steps} \
    --lr ${learning_rate} \
    --gamma ${discount_factor_gamma} \
    --ppo-epoch ${ppo_epochs} \
    --clip-params ${ppo_clip_param} \
    --max-grad-norm ${max_grad_norm_val} \
    --entropy-coef ${entropy_coef_val} \
    --num-mini-batch ${num_mini_batches} \
    --data-chunk-length ${data_chunk_len} \
    ${use_wandb_logging} \
    --wandb-name "${wandb_project_id}" \
    --user-name "${user_identifier}" \
    --render-mode "${render_mode}" # Added render mode to command line arguments

echo "--------------------------------------------------------------------------------------"
echo "Logic test execution finished."
echo "======================================================================================"