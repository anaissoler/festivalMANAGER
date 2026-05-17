"""
Configuration of the reward function's hyperparameters.
Modify this file to experiment without changing the logic.
"""

#  Constants
SAFETY_PENALTY = 1000.0   # Massive penalty if something dangerous happens 
SAFETY_VIOLATION_OCC = 0.95     # 95% occupancy is the red line for safety
CROWD_PENALTY_OCC = 0.85     # Over 85%, people start feeling uncomfortable
QUEUE_PENALTY_THRESH = 0.40     # Up to 40% queue is okay, more than that is a penalty
 
 

#  How much we care about each factor depending on the goal
MODE_WEIGHTS = {
    "A": {"rev": 0.8, "hap": 0.2, "queue": 0.3, "crowd": 0.3},
    "B": {"rev": 0.2, "hap": 0.8, "queue": 0.6, "crowd": 0.6},
    "C": {"rev": 0.5, "hap": 0.5, "queue": 0.5, "crowd": 0.5},
}