"""
Core training module for the Q-Learning agent in Festival.
Implements the tabular Q-Learning algorithm from scratch using only numpy
and Python standard library. Imports the Festival environment.

The workflow is:
    1. Build the environment for a given reward mode (A, B, or C)
    2. Determine the action space size from the environment
    3. Instantiate QLearningAgent with hyperparameters from agent_config.py
    4. Run the training loop (epsilon-greedy exploration + Q-table updates)
    5. Evaluate the trained agent greedily (epsilon=0)
    6. Save the Q-table (.npy) and reward log (.csv) 
"""

import os
import csv
import argparse
import numpy as np
from collections import defaultdict
 
from agent.agent_config import QLEARNING_CONFIG, TRAINING_CONFIG
from env.festival_env import FestivalEnv

# Wrappers
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, NormalizeObservation


# Instantiate a FestivalEnv for the given reward mode (A, B or C).
def _make_env(mode):
    env = FestivalEnv(mode=mode)
    env = TimeLimit(env, max_episode_steps=12)
    env = RecordEpisodeStatistics(env) 
    env = NormalizeObservation(env)
    return env
 

# Tabular Q-Learning agent
class QLearningAgent:
    def __init__(self, n_actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
 
        # Q[state] = array of n_actions values, initialised to 0.
        # defaultdict grows automatically as new states are visited.
        self.Q = defaultdict(lambda: np.zeros(n_actions))
 
    # Epsilon-greedy policy. If greedy=True always exploits (evaluation mode).
    def select_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # exploración
        return int(np.argmax(self.Q[state])) # explotación
    
    # Off-policy Q-Learning update. If the episode ends, the future term is set to 0.
    def update(self, state, action, reward, next_state, terminated):
        current_q = self.Q[state][action]
        future_q = 0.0 if terminated else np.max(self.Q[next_state])
        target = reward + self.gamma * future_q
        self.Q[state][action] += self.alpha * (target - current_q)
 
    # Multiply epsilon by epsilon_decay, flooring at epsilon_min.
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
 
    # Persist the Q-table to disk as a .npy file (allow_pickle=True).
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        q_dict = {str(k): v for k, v in self.Q.items()}
        np.save(filepath, q_dict, allow_pickle=True)
        print(f"  Q-table guardada en: {filepath}")
 
    # Class method: restore a previously saved Q-table from disk.
    @classmethod
    def load(cls, filepath, n_actions, **config):
        agent = cls(n_actions=n_actions, **config)
        q_dict = np.load(filepath, allow_pickle=True).item()
        agent.Q = defaultdict(
            lambda: np.zeros(n_actions),
            {int(k): v for k, v in q_dict.items()}
        )
        print(f"  Q-table cargada desde: {filepath}")
        return agent
 
 
# Helper functions

# Convert a continuous observation array into a single integer state key.
def _discretize(obs):
    obs = np.array(obs)

    bins = np.digitize(obs, bins=[-2, -1, 0, 1, 2, 3])

    result = 0
    for b in bins:
        result = result * 10 + int(b)

    return result
 
# Run one full episode and return the total undiscounted reward.
# If greedy=False the agent explores and updates its Q-table.
# If greedy=True  the agent exploits only (used for evaluation).
def _run_episode(agent, env, greedy=False):
    raw_state, _ = env.reset()
    state = _discretize(raw_state)
    total_reward = 0.0

    while True:
        action_idx = agent.select_action(state, greedy=greedy)
        if hasattr(env, "action_space") and hasattr(env.action_space, "nvec"):
            multi_action = []
            idx = action_idx
            for n in reversed(env.action_space.nvec):
                multi_action.append(idx % n)
                idx //= n
            action_for_env = list(reversed(multi_action))
        else:
            action_for_env = action_idx
        raw_next, reward, terminated, truncated, info = env.step(action_for_env)  # _ -> info
        next_state = _discretize(raw_next)
        done = terminated or truncated

        if not greedy:
            agent.update(state, action_idx, reward, next_state, terminated)

        total_reward += reward
        state = next_state

        if done:
            ep_stats = info.get("episode", {})  # solo existe al final del episodio
            break

    return total_reward, ep_stats
 
# Write the per-episode reward log to a CSV file  
def _save_log(rewards_per_episode, mode, results_path):
    os.makedirs(results_path, exist_ok=True)
    filepath = os.path.join(results_path, f"rewards_{mode}.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'epsilon'])
        for row in rewards_per_episode:
            writer.writerow(row)
    print(f"  Log guardado en: {filepath}")
 

# Public training functions

# Train a Q-Learning agent for a single reward mode (A, B or C).
# Saves qtable_{mode}.npy to ./models/ and rewards_{mode}.csv to ./results/.
# Returns the trained QLearningAgent instance.
def train_mode(mode):
    print(f"\n{'='*50}")
    print(f"Entrenando modo '{mode}' — Q-Learning tabular")
    print(f"{'='*50}")
 
    env = _make_env(mode)
    cfg = QLEARNING_CONFIG
    t_cfg = TRAINING_CONFIG
 
    # Determine n_actions from the real Gymnasium env or the stub
    if hasattr(env, 'action_space'):
        from gymnasium import spaces
        if isinstance(env.action_space, spaces.Discrete):
            n_actions = env.action_space.n
        elif isinstance(env.action_space, spaces.MultiDiscrete):
            # Flatten MultiDiscrete into a single linear index space
            n_actions = int(np.prod(env.action_space.nvec))
        else:
            raise ValueError(f"Unsupported action_space type: {type(env.action_space)}")
    else:
        n_actions = env.n_actions # stub fallback
 
    agent = QLearningAgent(
        n_actions = n_actions,
        alpha = cfg['alpha'],
        gamma = cfg['gamma'],
        epsilon = cfg['epsilon'],
        epsilon_min = cfg['epsilon_min'],
        epsilon_decay = cfg['epsilon_decay'],
    )
 
    rewards_log = [] # list of (episode, total_reward, epsilon)
 
    for ep in range(1, t_cfg['n_episodes'] + 1):
        ep_reward, ep_stats = _run_episode(agent, env)
        agent.decay_epsilon()

        reward_to_log = ep_stats.get("r", ep_reward)   # recompensa acumulada por el wrapper
        length_to_log = ep_stats.get("l", 0)            # cuántos slots duró el episodio
        rewards_log.append((ep, round(reward_to_log, 4), round(agent.epsilon, 6)))

        if ep % t_cfg['log_interval'] == 0:
            recent_mean = np.mean([r[1] for r in rewards_log[-t_cfg['log_interval']:]])
            print(f"  Ep {ep:5d}/{t_cfg['n_episodes']} | "
                f"Recompensa media (últimos {t_cfg['log_interval']}): {recent_mean:8.2f} | "
                f"ε={agent.epsilon:.4f} | "
                f"Slots: {length_to_log}")
 
 
    # Final greedy evaluation (no exploration)
    print(f"\n  Final evaluation ({t_cfg['eval_episodes']} episodes, ε=0):")
    eval_rewards = [_run_episode(agent, env, greedy=True)[0] for _ in range(t_cfg['eval_episodes'])]
    print(f"  Mean evaluation reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")    
 
    # Save Q-table and reward log
    save_path = t_cfg['save_path']
    agent.save(os.path.join(save_path, f"qtable_{mode}.npy"))
    _save_log(rewards_log, mode, t_cfg['results_path'])
 
    return agent
 
# Train one agent per reward mode (A, B, C) sequentially. Returns a dict {mode: QLearningAgent}.
def train_all_modes():
    agents = {}
    for mode in TRAINING_CONFIG['modes']:
        agents[mode] = train_mode(mode)
    print("\n Training completed for all modes.")
    return agents
 

 
# Entry point 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-Learning training — Festival Manager')
    parser.add_argument(
        '--mode',
        choices = ['A', 'B', 'C', 'all'],
        default = 'C',
        help    = 'Training mode: A, B, C (single mode) or all (all three)'
    )
    args = parser.parse_args()

    if args.mode == 'all':
        train_all_modes()
    else:
        train_mode(args.mode)