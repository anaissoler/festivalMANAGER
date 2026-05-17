"""
Configuration file for the Q-Learning agent used in Festival Manager RL.
It does not implement logic — it only defines the hyperparameters and
training settings that train.py will import and use.

The workflow of agent_config.py is:
    1. Define Q-Learning hyperparameters (alpha, gamma, epsilon schedule)
    2. Define training settings (episodes, modes, paths, logging intervals)
    3. Expose two dicts: QLEARNING_CONFIG and TRAINING_CONFIG
    4. Optionally verify the configuration by running this file directly
"""

# Hyperparameters of the Q-Learning algorithm
QLEARNING_CONFIG = dict(
    alpha = 0.1, # Learning rate: how much the Q-value is updated on each step (0=no learning, 1=overwrite completely)
    gamma = 0.99, # Discount factor: how much future rewards matter vs immediate ones (close to 1 = far-sighted agent)
    epsilon = 1.0, # Initial epsilon for ε-greedy exploration (1.0 = 100% random actions at the start)
    epsilon_min = 0.05, # Minimum epsilon floor: even after full training, 5% of actions remain random to avoid overfitting
    epsilon_decay = 0.995, # Multiplicative decay applied to epsilon after each episode (0.995 = slow, stable decay over ~2000 episodes)
)


# Training setup
TRAINING_CONFIG = dict(
    n_episodes = 2000, # Total number of training episodes — one full festival scheduling sequence per episode
    modes = ['A', 'B', 'C'], # Reward modes to train: A=revenue maximization, B=fan happiness, C=balanced objective
    save_path = './models/', # Directory where trained Q-tables will be saved as .npy files (one per mode)
    results_path = './results/', # Directory where reward logs will be saved as .csv files for later analysis 
    log_interval = 100, # How often (in episodes) a summary is printed to the console during training           
    eval_episodes = 20, # Number of greedy episodes (epsilon=0) run at the end to measure final agent performance            
)


# Quick verification -- print the two configuration dictionaries to the console
if __name__ == '__main__':
    print("QLEARNING_CONFIG:")
    for k, v in QLEARNING_CONFIG.items():
        print(f"  {k}: {v}")
    print("\nTRAINING_CONFIG:")
    for k, v in TRAINING_CONFIG.items():
        print(f"  {k}: {v}")