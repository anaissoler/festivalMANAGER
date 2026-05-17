"""
Festival's global constants file. It contains no logic or imports of its own—it's 
the central "dictionary" that all other modules import to access the festival's 
configuration.

The flow of config.py is:
    1. Define the festival layout (stages, sections, zones, artists)
    2. Define maximum capacities per zone
    3. Import and display reward weights per mode (A, B, C)
    4. Define safety and comfort thresholds
    5. Optionally verify the configuration by running this file directly
"""

import numpy as np
from env.reward_config import MODE_WEIGHTS


# Festival layout
N_STAGES = 3 # Number of stages: Main Stage, Stage B, Electronic Stage
N_SLOTS = 12 # Time slots per day: 12:00, 13:00, ... 23:00 (1 slot = 1 hour)
N_ZONES = 5 # Zones: stage0, stage1, stage2, bars, entrance
N_BARS = 5 # Number of bar areas in the festival
N_ARTISTS = 15 # Total artists available for scheduling

# Maximum capacity (people) per zone. Key = zone index.
#   0 → Main Stage : 5000
#   1 → Stage B : 2000
#   2 → Electronic : 1500
#   3 → Bars : 800
#   4 → Entrance : 3000
MAX_CAPACITY = {0: 5000, 1: 2000, 2: 1500, 3: 800, 4: 3000}


# Reward weights per operating mode
LAMBDA_A = MODE_WEIGHTS['A']
LAMBDA_B = MODE_WEIGHTS['B']
LAMBDA_C = MODE_WEIGHTS['C']
LAMBDAS = MODE_WEIGHTS


# Safety
# Catastrophic penalty applied when any zone exceeds 95% capacity
SAFETY_PENALTY = 1000.0

# Occupancy thresholds
COMFORT_THRESHOLD = 0.70 # Above this → comfort starts degrading
SAFETY_THRESHOLD = 0.95 # Above this → safety penalty is triggered


# Quick sanity check
if __name__ == '__main__':
    print(N_STAGES, N_SLOTS) # Expected output: 3 12
    assert len(MAX_CAPACITY) == N_ZONES
    assert all(0 < v <= 1 for v in LAMBDA_C.values())
    print("config.py OK")
