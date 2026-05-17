"""
Handles the reward function for the festival manager.
Basically, it decides if the agent is doing a good job or not.

The formula (from the project proposal) is:
    r_t = lambda1 * Revenue + lambda2 * Happiness - Penalties (Queues, Crowds, Safety)

We have 3 modes:
    - Mode A: Focus on Money (Revenue)
    - Mode B: Focus on People (Happiness)
    - Mode C: Balanced approach
"""

# Libraries 
import numpy as np

# Import from reward_config.py
from .reward_config import (
    MODE_WEIGHTS,
    SAFETY_PENALTY,
    SAFETY_VIOLATION_OCC,
    CROWD_PENALTY_OCC,
    QUEUE_PENALTY_THRESH,
)
 
 
#  Base reward function
def compute_reward(
    revenue,
    happiness,
    queue_len,
    zone_occ,
    lam,
    # Optional args for compatibility with the environment
    bar_demand=None,
    stage_occ=None,
    staff_levels=None,
    mode="C",
    revenue_so_far=0.0,
    max_revenue=1.0,
):

    # Determine which weights to use (A, B, or C)
    if isinstance(lam, str):
        assert lam in MODE_WEIGHTS, f"Modo invalido: {lam!r}"
        w = MODE_WEIGHTS[lam]
    else:
        w = lam
 
    # 1. Normalized revenue (0 to 1)
    R_revenue = float(np.clip(float(revenue) / max(float(max_revenue), 1e-6), 0.0, 1.0))
 
    # 2. Happiness (0 to 1)
    R_happiness = float(np.clip(happiness, 0.0, 1.0))
 
    # 3. Line penalty: how much do the lines exceed our 40% threshold?
    queue_excess = np.maximum(0.0, np.asarray(queue_len) - QUEUE_PENALTY_THRESH)
    P_queue = float(np.mean(queue_excess))
 
    # 4. Crowd penalty: are the zones getting too packed (>85%)?
    all_occ      = np.asarray(zone_occ)
    crowd_excess = np.maximum(0.0, all_occ - CROWD_PENALTY_OCC)
    P_crowd      = float(np.mean(crowd_excess))
 
    # 5. Safety Check: if any zone is over 95%, apply the catastrophic penalty
    safety_violation = bool(np.any(all_occ > SAFETY_VIOLATION_OCC))
    I_safety = SAFETY_PENALTY if safety_violation else 0.0
 
    # Calculate final reward based on the mode's weights
    reward = (
          w["rev"]   * R_revenue
        + w["hap"]   * R_happiness
        - w["queue"] * P_queue
        - w["crowd"] * P_crowd
        - I_safety
    )
 
    return float(reward)
 
 

# Extra bonuses to help the AI learn better and faster 
def compute_break_bonus(schedule_prev, schedule_curr, bar_demand_curr):
    """
    Strategic Break Bonus:
    If the AI clears the stages (no one playing) after a busy slot to push 
    people to the bars, we give it a small bonus. This is a key project concept.
    """
    prev_active = int(np.sum(np.asarray(schedule_prev) != -1))
    curr_active = int(np.sum(np.asarray(schedule_curr) != -1))

    # If we just went from 'music playing' to 'total silence'
    if prev_active > 0 and curr_active == 0:
        bar_avg = float(np.mean(bar_demand_curr))
        return float(np.clip(0.15 * bar_avg, 0.0, 0.15))
    return 0.0
 
 
def compute_staff_efficiency_bonus(bar_demand, queue_len, staff_levels):
    """
    Bonus for managing staff well:
    - Penalty if there are too many workers for very little demand (wasting money).
    - Bonus if demand is high but queues stay low (good management).
    """
    staff_norm = np.asarray(staff_levels, dtype=float) / 3.0
    demand = np.asarray(bar_demand)
    queue = np.asarray(queue_len)
 
    # Overstaffing penalty
    overstaff = np.maximum(0.0, staff_norm - demand)
    penalty   = -0.10 * float(np.mean(overstaff))
 
    # Good service bonus (high demand + short lines)
    well_served = np.where((demand > 0.5) & (queue < 0.3), 1.0, 0.0)
    bonus = 0.05 * float(np.mean(well_served))
 
    return float(np.clip(penalty + bonus, -0.1, 0.05))
 
 

#  Complete shaped reward
def compute_shaped_reward(
    happiness,
    bar_demand,
    queue_len,
    zone_occ,
    stage_occ,
    staff_levels,
    schedule_prev,
    schedule_curr,
    mode="C",
    revenue_so_far=0.0,
    max_revenue=1.0,
):

    assert mode in MODE_WEIGHTS, f"Modo invalido: {mode!r}"
    lam = MODE_WEIGHTS[mode]
 
    # Calculate simulated revenue for the slot based on bar demand and crowd happiness
    happiness_multiplier = 0.5 + 1.0 * float(happiness)
    raw_revenue = float(np.mean(bar_demand)) * happiness_multiplier
 
    # Prepare all components for the reward calculation
    R_revenue = float(np.clip(raw_revenue / max(float(max_revenue), 1e-6), 0.0, 1.0))
    R_happiness = float(np.clip(happiness, 0.0, 1.0))
    queue_excess = np.maximum(0.0, np.asarray(queue_len) - QUEUE_PENALTY_THRESH)
    P_queue = float(np.mean(queue_excess))
    all_occ = np.concatenate([np.asarray(stage_occ), np.asarray(zone_occ)])
    crowd_excess = np.maximum(0.0, all_occ - CROWD_PENALTY_OCC)
    P_crowd = float(np.mean(crowd_excess))
    safety_violation = bool(np.any(all_occ > SAFETY_VIOLATION_OCC))
    I_safety = SAFETY_PENALTY if safety_violation else 0.0
 
    # Base calculation
    base_reward = (
          lam["rev"] * R_revenue
        + lam["hap"] * R_happiness
        - lam["queue"] * P_queue
        - lam["crowd"] * P_crowd
        - I_safety
    )
 
    # Add the shaping bonuses
    break_bonus = compute_break_bonus(schedule_prev, schedule_curr, bar_demand)
    staff_bonus = compute_staff_efficiency_bonus(bar_demand, queue_len, staff_levels)
    shaped = base_reward + break_bonus + staff_bonus
 
    # Store everything in a dictionary to use it later in the logs/plots
    info = {
        "R_revenue": R_revenue,
        "R_happiness": R_happiness,
        "P_queue": P_queue,
        "P_crowd": P_crowd,
        "safety_violation": safety_violation,
        "I_safety": I_safety,
        "reward": float(base_reward),
        "shaped_reward": float(shaped),
        "break_bonus": break_bonus,
        "staff_bonus": staff_bonus,
        "mode": mode,
        "lambda1": lam["rev"],
        "lambda2": lam["hap"],
    }
    return float(shaped), info
 
 

# MAIN — manual execution to see how the reward function behaves with different inputs
if __name__ == "__main__":
    happiness = 0.72
    bar_demand = np.array([0.6, 0.5, 0.7, 0.4, 0.55])
    queue_len = np.array([0.3, 0.2, 0.5, 0.1, 0.25])
    zone_occ = np.array([0.7, 0.5, 0.4])
    stage_occ = np.array([0.8, 0.5, 0.3])
    staff_levels = np.array([2, 2, 3, 1, 2])
    sched_prev = np.array([0, 1, -1])    # habia artistas
    sched_curr = np.array([-1, -1, -1])  # descanso
 
    print("=" * 55)
    print("  Festival Manager - Demo Funcion de Recompensa")
    print("=" * 55)
    for mode in ["A", "B", "C"]:
        r, info = compute_shaped_reward(
            happiness, bar_demand, queue_len, zone_occ, stage_occ,
            staff_levels, sched_prev, sched_curr, mode=mode,
        )
        print(f"\n  Modo {mode} (lambda1={info['lambda1']}, lambda2={info['lambda2']}):")
        print(f"    R_revenue   = {info['R_revenue']:.3f}")
        print(f"    R_happiness = {info['R_happiness']:.3f}")
        print(f"    P_queue     = {info['P_queue']:.3f}")
        print(f"    P_crowd     = {info['P_crowd']:.3f}")
        print(f"    BreakBonus  = {info['break_bonus']:.3f}")
        print(f"    ─────────────────────────")
        print(f"    REWARD      = {r:.4f}")