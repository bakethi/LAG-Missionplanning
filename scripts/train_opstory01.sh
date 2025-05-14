#!/bin/sh

# Script to train the OpStory01 airbase attack scenario

# --- Configuration Variables ---
# The --env-name argument for train_jsbsim.py (e.g., "SingleCombat" if using SingleCombatEnv)
env_type="SingleCombat"

# The --scenario-name argument for train_jsbsim.py
# This points to your YAML config file, relative to envs/JSBSim/configs/
# Example: if OpStory01.yaml is in envs/JSBSim/configs/OpStory01Scenarios/OpStory01.yaml
# then scenario_config_path="OpStory01Scenarios/OpStory01"
# If OpStory01.yaml is directly in envs/JSBSim/configs/OpStory01.yaml
# then scenario_config_path="OpStory01"
scenario_config_path="OpStories/OpStory01" # MODIFY THIS AS NEEDED

# RL algorithm to use
rl_algorithm="ppo"

# A descriptive name for this experiment run
experiment_tag="OpStory01_AirbaseAttack_PPO_Run1"

# Random seed for reproducibility
random_seed=1

# Training resources
num_training_threads=1    # Typically 1 for the main training loop
num_rollout_threads=8     # Number of parallel environments for data collection (adjust to your CPU cores)
use_cuda_flag="--cuda"    # Use "--cuda" to enable GPU, or "" to disable (use CPU)

# Training duration and PPO hyperparameters
total_env_steps=10000000  # Total number of environment interactions
ppo_epochs=10             # Number of PPO epochs per update
num_mini_batches=1        # Number of mini-batches for PPO updates
data_chunk_len=10         # Length of data chunks for recurrent policies (if used)
learning_rate=5e-4
discount_factor_gamma=0.99
ppo_clip_param=0.2
max_grad_norm_val=2.0
entropy_coef_val=1e-3

# Logging and Saving
log_update_interval=1     # Log every N updates
save_model_interval=10    # Save model checkpoint every N updates
use_wandb_logging="--use-wandb" # Use "--use-wandb" to enable, or "" to disable
wandb_project_id="your_wandb_project_name" # Your project name on Weights & Biases
user_identifier="your_username"            # Your username for organizing results/W&B

# --- End Configuration Variables ---

# Inform the user about the settings
echo "======================================================================================"
echo "Starting Training for Operational Story 01 (Airbase Attack)"
echo "--------------------------------------------------------------------------------------"
echo "Environment Type         : ${env_type}"
echo "Scenario Config Path     : ${scenario_config_path}"
echo "RL Algorithm             : ${rl_algorithm}"
echo "Experiment Tag           : ${experiment_tag}"
echo "Random Seed              : ${random_seed}"
echo "Rollout Threads          : ${num_rollout_threads}"
echo "Total Environment Steps  : ${total_env_steps}"
echo "Using CUDA               : ${use_cuda_flag:-"No (CPU)"}" # Shows "No (CPU)" if use_cuda_flag is empty
echo "W&B Logging              : ${use_wandb_logging:-"No"}"  # Shows "No" if use_wandb_logging is empty
echo "======================================================================================"

# Construct the path to the training script
# Assumes this script is in LAG-Missionplanning/scripts/
# and train_jsbsim.py is in LAG-Missionplanning/scripts/train/
current_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
training_script_path="${current_script_dir}/train/train_jsbsim.py"

# Check if the training script exists
if [ ! -f "${training_script_path}" ]; then
    echo "ERROR: Training script not found at ${training_script_path}"
    echo "Please ensure this script is in the 'LAG-Missionplanning/scripts/' directory."
    exit 1
fi

# Execute the training script
# Add any other necessary flags from the original train_jsbsim.py or other examples
# e.g., --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 etc.
# For a non-self-play scenario, remove --use-selfplay and related args.

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
    # Add other relevant PPO/model parameters as found in train_selfplay.sh if needed:
    # --hidden-size "128 128" \
    # --act-hidden-size "128 128" \
    # --recurrent-hidden-size 128 \
    # --recurrent-hidden-layers 1 \
    # For evaluation during training (optional):
    # --use-eval --n-eval-rollout-threads 1 --eval-interval 50 --eval-episodes 5

echo "--------------------------------------------------------------------------------------"
echo "Training script execution finished."
echo "======================================================================================"