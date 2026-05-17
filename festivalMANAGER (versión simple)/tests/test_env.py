"""
Verify that the FestivalEnv environment is working correctly before training the agent

The 6 tests performed are:
    1. test_env_reset
    2. test_env_step_valid
    3. test_full_episode_length
    4. test_safety_penalty
    5. test_check_env
    6. test_all_modes

"""
# Libraries 
import numpy as np

# Import from other modules
from gymnasium.utils.env_checker import check_env
from env.festival_env import FestivalEnv
from agent.config import N_ZONES, N_SLOTS
from env.reward_config import SAFETY_PENALTY
from env.festival_env import obs_dim



# HELPERS for tests
# Create a new environment instance for each test to ensure independence
def make_env(mode='C'):
    return FestivalEnv(mode=mode)



# TESTS
# 1. Reset test: Verify that resetting the environment returns an observation 
# with the correct format, type, and range
def test_env_reset():
    env = make_env()
    obs, info = env.reset(seed=42)

    assert obs.shape == (obs_dim,), f"Expected shape ({obs_dim},), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert obs.min() >= 0.0, f"Observation below 0: {obs.min()}"
    assert obs.max() <= 1.0, f"Observation above 1: {obs.max()}"
    assert isinstance(info, dict)

# 2. Step test: Verifies that when the agent takes an action, the environment returns 
# exactly what it should: observation, reward, terminated, truncated, and info with the correct types
def test_env_step_valid():
    env = make_env()
    env.reset(seed=0)

    action = env.action_space.sample()
    result = env.step(action)

    assert len(result) == 5, "step() must return (obs, reward, terminated, truncated, info)"
    obs, reward, terminated, truncated, info = result

    assert obs.shape == (obs_dim,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert obs.min() >= 0.0 and obs.max() <= 1.0


# 3. Full episode test: Verifies that a full episode lasts exactly N_SLOTS steps 
# (in this case, 12, as defined in config.py)
def test_full_episode_length():
    env = make_env()
    env.reset(seed=1)

    step_count = 0
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_count += 1

    assert step_count == N_SLOTS, f"Expected {N_SLOTS} steps, got {step_count}"

# 4. Safety penalty test: Verifies that when a zone is at 100% occupancy, 
# the safety penalty is triggered and the reward drops below -SAFETY_PENALTY/2
def test_safety_penalty():
    env = make_env()
    env.reset(seed=2)

    for zone_idx in range(N_ZONES):
        env.zone_occ[zone_idx] = 1.0

    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)

    assert reward < -SAFETY_PENALTY/2, (
        f"Safety penalty should yield reward < {-SAFETY_PENALTY/2:.2f}, got {reward:.2f}"
    )

# 5. Environment check: Use Gymnasium's check_env function to verify that 
# the environment complies with the standard Gymnasium API
def test_check_env():
    env = make_env()
    check_env(env)  

# 6. Mode Test: Verify that modes A, B, and C all function properly
def test_all_modes():
    for mode in ['A', 'B', 'C']:
        env = FestivalEnv(mode=mode)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (obs_dim,), f"Mode {mode}: wrong obs shape"
        action = env.action_space.sample()
        obs2, reward, _, _, _ = env.step(action)
        assert obs2.shape == (obs_dim,)
        assert isinstance(reward, float)
