"""
This is the heart of the project: the Festival Manager environment.

Instead of directly using Gymnasium’s predefined environment,
we have created our own, following the same API structure.

We have implemented:
  - MultiDiscreteSpace (for structured discrete actions)
  - BoxSpace (for continuous observations in [0,1])

Observation space:
    A normalized vector in the range [0,1] that includes:
    - The current schedule (flattened)
    - Occupancy levels of zones and stages
    - Bar line lengths
    - Overall happiness
    - Cumulative revenue
    - Temporal progression

Action space:
    A MultiDiscrete structure representing:
    [artist, stage, time slot, staffing level]

    - artist: which performer to schedule
    - stage: where they perform
    - time slot: when they perform
    - staffing level: intensity of bar staffing

The environment simulates the dynamics of the festival using a crowd simulator
and evaluates decisions using a reward function that balances:
    - revenue
    - user satisfaction (happiness)
    - safety constraints
"""

# Libraries
import numpy as np

# Importing our custom project components
from .data_generator  import generate_festival_config
from .crowd_simulator import CrowdSimulator
from .reward_function import compute_shaped_reward
from .reward_config import SAFETY_VIOLATION_OCC


# Custom action space — replaces gymnasium.spaces.MultiDiscrete
# Represents a tuple of independent discrete dimensions
# In our case: [16 artists, 3 stages, 12 slots, 4 staff levels]
class MultiDiscreteSpace:
    def __init__(self, nvec, seed=None):
        self.nvec = np.array(nvec, dtype=np.int64) # limits of each dimension
        self.shape = (len(self.nvec),)
        self.dtype = np.int64
        self._rng = np.random.default_rng(seed) # random generator for sampling

    def sample(self):
        # Returns a random valid action, one integer per dimension
        # e.g. [7, 1, 5, 2] → artist 7, stage 1, slot 5, staff level 2
        return self._rng.integers(0, self.nvec, dtype=np.int64)

    def contains(self, action):
        action = np.asarray(action)
        # Checks the action has exactly 4 elements
        if action.shape != self.shape:
            return False
        
        # Checks that each value is within its valid range
        all_in_range = all(0 <= action[i] < self.nvec[i] for i in range(len(self.nvec)))
        return all_in_range


# Custom observation space — replaces gymnasium.spaces.Box
# Represents a continuous n-dimensional space with values in [low, high]
class BoxSpace:
    def __init__(self, low, high, shape, dtype=np.float32, seed=None):
        self.low = float(low) # low is float → 0.0
        self.high = float(high) # high is float → 1.0
        self.shape = tuple(shape)
        self.dtype = dtype
        self._rng = np.random.default_rng(seed) # random generator for sampling

    def sample(self):
        # Returns a random observation uniformly sampled from [0.0, 1.0]
        return self._rng.uniform(
            self.low, self.high, size=self.shape
        ).astype(self.dtype)

    def contains(self, obs):
        # Checks that the observation has the right shape and all values are in [0.0, 1.0]
        obs = np.asarray(obs)
        return (
            obs.shape == self.shape
            and bool(np.all(obs >= self.low))
            and bool(np.all(obs <= self.high))
        )



class FestivalEnv():
    """
    Music Festival Planning Environment.

    The Agent acts as the Operations Director and must decide:
      - The lineup (who plays where and when).
      - How much staff to put in the bars.

    Goal: Balance money (revenue), fan happiness, and safety.
    """

    # Gymnasium metadata
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        n_stages=3,
        n_slots=12,
        n_artists=15,
        n_bars=5,
        total_attendees=8000,
        mode="C",
        seed=42,
        noise_std=0.02,
        render_mode=None,
    ):
        super().__init__()

        # Basic checks to avoid weird errors later
        assert mode in ("A", "B", "C"), f"Modo invalido: {mode!r}. Usa 'A', 'B' o 'C'."
        assert n_stages >= 2, "Se necesitan al menos 2 escenarios."
        assert n_slots  >= 4, "Se necesitan al menos 4 slots."

        self.n_stages = n_stages
        self.n_slots = n_slots
        self.n_artists = n_artists
        self.n_bars = n_bars
        self.mode = mode
        self.noise_std = noise_std
        self.render_mode = render_mode

        # Generate the festival setup (stages, artists names, etc.)
        self.config = generate_festival_config(
            n_stages=n_stages,
            n_slots=n_slots,
            n_artists=n_artists,
            n_bars=n_bars,
            total_attendees=total_attendees,
            seed=seed if seed is not None else 42,
        )
        self.n_zones = self.config["n_zones"]

        # Action Space: MultiDiscrete 
        # As seen in notes (page 46): "Collection of several discrete spaces"
        # Each action is a list: [artist_idx, stage_idx, slot_idx, staff_level]
        self.action_space = MultiDiscreteSpace([
            n_artists + 1,   # Artist (0 to n-1) or "n" for empty slot
            n_stages,        # Which stage
            n_slots,         # Which hour
            4,               # Level of staff working (0=Low, 3=Max)
        ])

        # Observation Space: Continuous Box [0, 1]
        # As seen in notes (page 46): "Continuous space with upper and lower limits"
        # We flatten everything into a single 1D array so the AI can read it easily
        obs_dim = (
            n_stages * n_slots   # Flattened schedule
            + self.n_zones       # How full each zone is
            + n_stages           # How full each stage is
            + n_bars             # How long the bar lines are
            + 1                  # General happiness
            + 1                  # Current money
            + 1                  # How much time is left
        )
        self.observation_space = BoxSpace(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # The simulator that handles the "physics" of the crowd
        self._sim = CrowdSimulator(self.config, seed=seed if seed is not None else 0,
                                   noise_std=noise_std)

        # Set the festival to the starting state
        self._reset_state()

    # Reset: mandatory gymnasium method
    def reset(self, seed=None, options=None):
        
        # Restarts the environment to the beginning of the day.

        if seed is not None:
            self._sim = CrowdSimulator(
                self.config, seed=seed, noise_std=self.noise_std)

        self._reset_state()
        return self._build_obs(), self._build_info()

    def _reset_state(self):
        # Helper to clear all variables at the start of a new episode.
        # Schedule: -1 means "nothing planned yet"
        self.schedule_matrix = -np.ones((self.n_stages, self.n_slots), dtype=np.int32)

        # Staff levels start at 1 by default 
        self.staff_matrix = np.ones((self.n_bars, self.n_slots), dtype=np.int32)

        # Initial crowd stats (empty stages, 60% happiness)
        self.stage_occ  = np.zeros(self.n_stages, dtype=np.float32)
        self.zone_occ   = np.zeros(self.n_zones,  dtype=np.float32)
        self.bar_demand = np.zeros(self.n_bars,   dtype=np.float32)
        self.queue_len  = np.zeros(self.n_bars,   dtype=np.float32)
        self.happiness  = 0.60   # felicidad inicial moderada

        # Ingresos y control del episodio
        self.revenue_so_far = 0.0
        self.revenue_max_est = float(self.n_slots * self.n_bars * 0.8)
        self.current_slot = 0
        self.episode_reward = 0.0
        self.safety_violations = 0

    # Step: mandatory gymnasium method
    def step(self, action):
        
        # Execute one action: place an artist, set staff, and see what happens.
        assert self.action_space.contains(action), f"Accion invalida: {action}"

        artist_idx = int(action[0])
        stage_idx = int(action[1])
        slot_idx = int(action[2])
        staff_lvl  = int(action[3])

        # Apply the decision to our schedule
        real_artist = artist_idx if artist_idx < self.n_artists else -1
        self.schedule_matrix[stage_idx, slot_idx]  = real_artist
        self.staff_matrix[:, slot_idx]             = staff_lvl

        # Check what was happening before vs what is happening now 
        schedule_prev = self.schedule_matrix[:, max(0, self.current_slot - 1)]
        schedule_curr = self.schedule_matrix[:, self.current_slot]
        staff_curr    = self.staff_matrix[:, self.current_slot].astype(float)

        # 1. Update the crowd simulation for this time step
        time_step  = self.current_slot / max(self.n_slots - 1, 1)
        sim_result = self._sim.step(
            schedule_row=schedule_curr,
            staff_levels=staff_curr,
            prev_stage_occ=self.stage_occ,
            prev_zone_occ=self.zone_occ,
            prev_happiness=self.happiness,
            time_step=time_step,
        )

        # 2. Update our local variables with the simulation results
        self.stage_occ = sim_result["stage_occ"]
        self.zone_occ = sim_result["zone_occ"]
        self.bar_demand = sim_result["bar_demand"]
        self.queue_len = sim_result["queue_len"]
        self.happiness = sim_result["happiness"]

        # 3. Calculate the reward based on Mode A, B, or C
        reward, reward_info = compute_shaped_reward(
            happiness=self.happiness,
            bar_demand=self.bar_demand,
            queue_len=self.queue_len,
            zone_occ=self.zone_occ,
            stage_occ=self.stage_occ,
            staff_levels=staff_curr,
            schedule_prev=schedule_prev,
            schedule_curr=schedule_curr,
            mode=self.mode,
            revenue_so_far=self.revenue_so_far,
            max_revenue=self.revenue_max_est,
        )

        # Acumulate revenue so far
        self.revenue_so_far += reward_info["R_revenue"]
        self.episode_reward += reward

        if reward_info["safety_violation"]:
            self.safety_violations += 1

        # Move to the next hour
        self.current_slot += 1

        # Check if the festival is over
        # - truncated: end of the day reached
        # - terminated: too many safety alerts 
        truncated = self.current_slot >= self.n_slots
        terminated = self.safety_violations >= 3

        # Final bonus for finishing the day with happy people
        if truncated and not terminated:
            reward += self.happiness * 2.0
            self.episode_reward += self.happiness * 2.0

        info = self._build_info()
        info.update(reward_info)

        if self.render_mode == "ansi":
            self.render()

        return self._build_obs(), float(reward), terminated, truncated, info

    # Observation builder
    def _build_obs(self):
        
        # Creates the [0, 1] vector the AI uses to 'see' the environment.
        # We normalize everything so the AI doesn't get confused by big numbers.
        
        # Normalize schedule (-1 to 0, artist IDs to 0.0-1.0)
        sched_raw  = self.schedule_matrix.astype(np.float32)
        sched_norm = np.where(sched_raw == -1, 0.0, sched_raw / self.n_artists)
        sched_norm = np.clip(sched_norm, 0.0, 1.0).flatten()

        # Normalize revenue using a simple sigmoid so it stays between 0 and 1
        rev_raw  = self.revenue_so_far / max(self.revenue_max_est, 1e-6)
        rev_norm = float(np.clip(rev_raw / (1.0 + rev_raw), 0.0, 1.0))

        obs = np.concatenate([
            sched_norm,                                                    # (n_stages*n_slots,)
            np.clip(self.zone_occ,  0.0, 1.0).astype(np.float32),         # (n_zones,)
            np.clip(self.stage_occ, 0.0, 1.0).astype(np.float32),         # (n_stages,)
            np.clip(self.queue_len, 0.0, 1.0).astype(np.float32),         # (n_bars,)
            np.array([float(np.clip(self.happiness, 0.0, 1.0))],
                     dtype=np.float32),                                    # (1,)
            np.array([rev_norm], dtype=np.float32),                        # (1,)
            np.array([min(1.0, self.current_slot / max(self.n_slots-1,1))],
                     dtype=np.float32),                                    # (1,)
        ], dtype=np.float32)

        return obs

    # Info builder

    def _build_info(self):
        return {
            "current_slot":      self.current_slot,
            "happiness":         float(self.happiness),
            "revenue_so_far":    float(self.revenue_so_far),
            "safety_violations": self.safety_violations,
            "episode_reward":    float(self.episode_reward),
            "mode":              self.mode,
        }

    # Render
    def render(self):
    
        # Prints a text-based dashboard of the current festival status.
        if self.render_mode != "ansi":
            return None

        slot = self.current_slot
        lines = [
            f"+-- Festival Manager -- Slot {slot:02d}/{self.n_slots} -- Modo {self.mode} --+",
        ]
        lines.append("|  HORARIO ACTUAL:")
        for s_idx in range(self.n_stages):
            aid   = int(self.schedule_matrix[s_idx, max(0, slot - 1)])
            name  = (self.config["artists"][aid]["name"]
                     if aid != -1 else "-- DESCANSO --")
            sname = self.config["stages"][s_idx]["name"]
            lines.append(f"|    {sname:25s} -> {name}")

        lines.append("|  METRICAS:")
        occ_str = "  ".join(f"E{i}:{o*100:.0f}%" for i, o in enumerate(self.stage_occ))
        lines.append(f"|    Ocupacion  : {occ_str}")
        q_str = "  ".join(f"B{i}:{q:.2f}" for i, q in enumerate(self.queue_len))
        lines.append(f"|    Colas      : {q_str}")
        bar_f = "#" * int(self.happiness * 20)
        lines.append(f"|    Felicidad  : [{bar_f:20s}] {self.happiness*100:.1f}%")
        lines.append(f"|    Ingresos   : {self.revenue_so_far:.3f}"
                     f"  |  Alertas: {self.safety_violations}")
        lines.append("+" + "-" * 60 + "+")

        output = "\n".join(lines)
        print(output)
        return output

    # Utilities
    def get_schedule_summary(self):
        # Handy function to get the whole timetable in a readable dict.
        summary = {}
        for s_idx in range(self.n_stages):
            stage_name = self.config["stages"][s_idx]["name"]
            slots_list = []
            for slot in range(self.n_slots):
                aid = int(self.schedule_matrix[s_idx, slot])
                slots_list.append({
                    "slot":   slot,
                    "artist": (self.config["artists"][aid]["name"]
                               if aid != -1 else "DESCANSO"),
                    "pop":    (self.config["artists"][aid]["popularity"]
                               if aid != -1 else 0.0),
                })
            summary[stage_name] = slots_list
        return summary

    def set_mode(self, mode):
        # Allows switching the goal (A, B, or C) mid-training if needed.
        assert mode in ("A", "B", "C"), f"Modo invalido: {mode!r}"
        self.mode = mode

    def close(self):
        # Cleanup resources
        pass




# MAIN — manual execution to test the environment
if __name__ == "__main__":
    print("Creando entorno FestivalEnv...")
    env = FestivalEnv(
        n_stages=3, n_slots=12, n_artists=15, n_bars=5,
        mode="C", render_mode="ansi", seed=42,
    )

    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space nvec: {env.action_space.nvec}")
    print()

    # Ciclo basico de RL, tal como aparece en los apuntes (pag. 41):
    # 1. env.reset()
    # 2. env.step(action) en bucle
    obs, info = env.reset()
    print(f"  Obs inicial shape: {obs.shape} | en [0,1]: {obs.min():.3f} - {obs.max():.3f}")

    total_reward = 0.0
    for step_n in range(env.n_slots):
        action = env.action_space.sample()   # accion aleatoria (como en apuntes)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step_n % 4 == 0:
            env.render()
        if terminated or truncated:
            break

    print(f"\n  Episodio completado en {env.current_slot} slots")
    print(f"  Recompensa total : {total_reward:.3f}")
    print(f"  Felicidad final  : {info['happiness']:.3f}")
    print(f"  Alertas seguridad: {info['safety_violations']}")
    env.close()