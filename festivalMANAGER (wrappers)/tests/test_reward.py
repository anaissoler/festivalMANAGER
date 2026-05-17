"""
Unit tests for the reward function, covering safety penalties, revenue, happiness, queue effects, and different configuration modes

It contains:
    1. test_reward_returns_float
    2. test_safety_penalty_applied
    3. test_no_safety_penalty_in_normal_conditions
    4. test_higher_happiness_increases_reward
    5. test_higher_revenue_increases_reward
    6. test_longer_queues_decrease_reward
    7. test_mode_a_weights_revenue_more
    8. test_mode_b_weights_happiness_more
    9. test_reward_range_without_safety
"""


# Libraries
import numpy as np

# Import from other modules
from env.reward_function import compute_reward
from agent.config import LAMBDA_A, LAMBDA_B, LAMBDA_C, N_BARS, N_ZONES, MAX_CAPACITY
from env.reward_config import SAFETY_PENALTY



# HELPERS for tests
# These functions create specific occupancy and queue scenarios to test the reward function's behavior under different conditions

# Normal occupancy: 40% of max capacity in all zones
def normal_occupancy():
    return np.array([
        MAX_CAPACITY[z] * 0.4 for z in range(N_ZONES)
    ], dtype=np.float32)

# Unsafe occupancy: 97% of max capacity in all zones, which should trigger the safety penalty
def unsafe_occupancy():
    return np.array([
        MAX_CAPACITY[z] * 0.97 for z in range(N_ZONES)
    ], dtype=np.float32)

# Zero queues: No waiting at the bars
# It permits to isolate the effect of other variables on the reward
def zero_queues():
    return np.zeros(N_BARS, dtype=np.float32)




# TESTS

# Verify that the reward function returns a float value
def test_reward_returns_float():
    r = compute_reward(10000.0, 0.8, zero_queues(), normal_occupancy(), LAMBDA_C)
    assert isinstance(r, float), f"Expected float, got {type(r)}"

# Verify that the safety penalty is applied when occupancy exceeds the safety threshold
def test_safety_penalty_applied():
    r = compute_reward(10000.0, 0.8, zero_queues(), unsafe_occupancy(), LAMBDA_C)
    assert r < -SAFETY_PENALTY/2, f"Safety penalty not applied — reward was {r:.2f}"

#  Verify that the safety penalty is NOT applied when occupancy is below the safety threshold
def test_no_safety_penalty_in_normal_conditions():
    r = compute_reward(10000.0, 0.8, zero_queues(), normal_occupancy(), LAMBDA_C)
    assert r > -SAFETY_PENALTY/2, f"Unexpected safety penalty — reward was {r:.2f}"

# Verify that a higher happiness score leads to a higher reward, all else being equal
def test_higher_happiness_increases_reward():
    occ = normal_occupancy()
    q = zero_queues()
    r_low = compute_reward(5000.0, 0.2, q, occ, LAMBDA_C)
    r_high = compute_reward(5000.0, 0.9, q, occ, LAMBDA_C)
    assert r_high > r_low, (
        f"Higher happiness should give higher reward: {r_high:.3f} <= {r_low:.3f}"
    )

# Verify that higher revenue leads to a higher reward, all else being equal
def test_higher_revenue_increases_reward():
    occ = normal_occupancy()
    q = zero_queues()
    r_low  = compute_reward(0.0,     0.5, q, occ, LAMBDA_C)
    r_high = compute_reward(40000.0, 0.5, q, occ, LAMBDA_C)
    assert r_high > r_low, (
        f"Higher revenue should give higher reward: {r_high:.3f} <= {r_low:.3f}"
    )

# Verify that longer queues lead to a lower reward, all else being equal
def test_longer_queues_decrease_reward():
    occ = normal_occupancy()
    q_short = np.zeros(N_BARS, dtype=np.float32)
    q_long = np.array([300.0, 300.0, 300.0, 300.0], dtype=np.float32)
    r_short = compute_reward(10000.0, 0.7, q_short, occ, LAMBDA_C)
    r_long = compute_reward(10000.0, 0.7, q_long,  occ, LAMBDA_C)
    assert r_short > r_long, (
        f"Long queues should lower reward: {r_short:.3f} <= {r_long:.3f}"
    )

# Check that mode A rewards revenue more than mode B, all else being equal
def test_mode_a_weights_revenue_more():
    occ = normal_occupancy()
    q = zero_queues()
    r_a = compute_reward(40000.0, 0.3, q, occ, LAMBDA_A)
    r_b = compute_reward(40000.0, 0.3, q, occ, LAMBDA_B)
    assert r_a > r_b, (
        f"Mode A should reward high revenue more than mode B: {r_a:.3f} <= {r_b:.3f}"
    )

# Check that mode B rewards happiness more than mode A, all else being equal
def test_mode_b_weights_happiness_more():
    occ = normal_occupancy()
    q = zero_queues()
    r_a = compute_reward(1000.0, 0.95, q, occ, LAMBDA_A)
    r_b = compute_reward(1000.0, 0.95, q, occ, LAMBDA_B)
    assert r_b > r_a, (
        f"Mode B should reward high happiness more than mode A: {r_b:.3f} <= {r_a:.3f}"
    )

# Ensure that the reward remains within a reasonable range (for example, between -5 and +2) under normal conditions without security breaches
def test_reward_range_without_safety():
    occ = normal_occupancy()
    q = zero_queues()
    r = compute_reward(10000.0, 0.7, q, occ, LAMBDA_C)
    assert -5.0 < r < 2.0, f"Reward out of expected range: {r:.4f}"
