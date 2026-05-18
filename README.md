#  Festival Manager — Reinforcement Learning Project

> **GitHub:** [https://github.com/anaissoler/festivalMANAGER](https://github.com/anaissoler/festivalMANAGER)

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

Results are saved automatically in `results/` and `models/`. Figures are saved in `results/figuras/` as `.html` and `.png` files.

---

## What is this project?

Festival Manager is a Reinforcement Learning project where we train an AI agent to **plan and manage a music festival**.

The agent takes on the role of the festival's **operations director**: at each time slot it decides which artist plays on which stage and how many bar staff to deploy. The challenge is that these decisions have cascading consequences — a popular artist fills the stage but empties the bars, and a well-timed break generates bar revenue but drops happiness if used too often.

The goal is to find the optimal balance between three things that are constantly in tension:

- **Revenue** — bars generate income, but only when people are not watching a show. Placing a break right after a popular artist creates a natural spending window where the crowd flows to the bars.
- **Fan happiness** — a great lineup, comfortable crowd density, and short queues keep attendees happy. A happy crowd spends more money, so happiness and revenue reinforce each other when managed well.
- **Safety** — no zone can exceed 95% occupancy. Three safety violations end the episode early, simulating a festival being shut down by authorities.

The reward signal that guides the agent at each step follows this formula:

```
r = λ1 × Revenue  +  λ2 × Happiness  −  λ3 × Queue penalty  −  λ4 × Crowd penalty  −  Safety penalty
```

The λ weights change depending on the chosen training mode, which lets us train three different agents with different priorities and compare what strategies they develop:

| Mode | Strategy | What the agent learns to prioritise |
|------|----------|-------------------------------------|
| **A** | Maximum Revenue | Schedule breaks after popular shows to push people to bars |
| **B** | Maximum Happiness | Keep crowds comfortable and queues short, even at a revenue cost |
| **C** | Balanced *(recommended)* | Find the sustainable combination of both goals |

On top of the base reward, the agent also receives two shaping bonuses designed to help it discover key festival management strategies faster:
- A **break bonus** when it clears all stages right after a busy slot, driving bar demand.
- A **staff efficiency bonus** for deploying the right number of bar staff relative to demand, without overstaffing.

---

## What all four versions share

All versions share the same core structure and files. The four folders listed below each contain these modules — what changes between versions is described in the next section.

```
festival_manager/
│
├── env/
│   ├── data_generator.py       ← Generates artists, stages, bars, zones
│   ├── crowd_simulator.py      ← Simulates crowd movement each time slot
│   ├── reward_function.py      ← Base reward formula + shaping bonuses
│   ├── reward_config.py        ← All reward constants and mode weights
│   ├── festival_env.py         ← The main RL environment
│   └── __init__.py
│
├── agent/
│   ├── config.py               ← Central layout constants (stages, slots, etc.)
│   ├── agent_config.py         ← Q-Learning hyperparameters and training settings
│   └── train.py                ← Tabular Q-Learning training loop
│
├── evaluation/
│   ├── evaluate.py             ← Runs trained agents and random baseline
│   └── visualizar.py           ← Generates all charts and figures
│
├── tests/
│   ├── test_env.py             ← Environment correctness tests
│   └── test_reward.py          ← Reward function unit tests
│
├── results/                    ← generated automatically on first run
├── models/                     ← generated automatically on first run
│
├── main.py                     ← runs everything in order
└── requirements.txt
```

### `env/reward_config.py` — reward constants (present from Version 1)

All reward constants and mode weights live in a dedicated file, separate from the reward logic. This encapsulation means any future tuning only requires changing one place. The constants it holds are:

```python
SAFETY_PENALTY        # -1000 applied when any zone exceeds 95% occupancy
SAFETY_VIOLATION_OCC  # 0.95 — the occupancy threshold that triggers the penalty
CROWD_PENALTY_OCC     # 0.85 — occupancy above which crowd comfort drops
QUEUE_PENALTY_THRESH  # 0.40 — queue length above which penalty is applied
MODE_WEIGHTS          # λ weights for Revenue, Happiness, Queue, Crowd per mode
```

All other files (`reward_function.py`, `festival_env.py`, `test_env.py`, `test_reward.py`) import these constants from `reward_config.py` only — never from `reward_function.py`.

### `agent/train.py` — tabular Q-Learning

Implements tabular Q-Learning from scratch using a `defaultdict` as the Q-table. Any unseen state is initialised to zero automatically. Runs the epsilon-greedy training loop for 2000 episodes per mode, saves the trained Q-table as a `.npy` file in `models/`, and logs per-episode rewards to a `.csv` file in `results/`. This is the file that changes most across versions.

### `evaluation/`

**`evaluate.py`** loads the saved Q-tables and runs evaluation episodes with epsilon set to 0. Also runs a random baseline for comparison. Saves all metrics to CSV files.

**`visualizar.py`** generates all figures: training curves per mode, Q-Learning vs random baseline comparison, bar charts of average metrics, a revenue–happiness scatter plot, and a heatmap of the schedule the agent learned for each mode.

### Q-Learning Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `alpha` | 0.1 | Learning rate — how much each update shifts the Q-value |
| `gamma` | 0.99 | Discount factor — the agent values future rewards heavily |
| `epsilon` | 1.0 → 0.05 | Exploration rate — starts fully random, gradually exploits |
| `epsilon_decay` | 0.995 | Decay per episode — converges after ~2000 episodes |
| `n_episodes` | 2000 | Total training episodes per mode |

---

## Project versions

The project was built incrementally across four versions. Each section below describes **only what changed** from the previous version.

---

### Version 1 — Simple (`festivalMANAGER (versión simple)`)

This is the starting point. The environment uses **Gymnasium's built-in library** directly, following the standard RL API as seen in the course notes.

`env/festival_env.py` inherits from `gym.Env` and uses Gymnasium's own space classes:

```python
import gymnasium as gym
from gymnasium import spaces

class FestivalEnv(gym.Env):
    ...
    self.action_space = spaces.MultiDiscrete([n_artists + 1, n_stages, n_slots, 4])
    self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
```

`agent/train.py` detects the action space type using Gymnasium's `isinstance` checks:

```python
from gymnasium import spaces
if isinstance(env.action_space, spaces.Discrete):
    n_actions = env.action_space.n
elif isinstance(env.action_space, spaces.MultiDiscrete):
    n_actions = int(np.prod(env.action_space.nvec))
```

The `_discretize()` function converts the continuous `[0, 1]` observation vector into a single integer state key by multiplying every feature by 10 and rounding down (10 buckets per feature):

```python
def _discretize(obs):
    bins = np.clip((obs * 10).astype(int), 0, 9)
    result = 0
    for b in bins:
        result = result * 10 + int(b)
    return result
```

`tests/test_env.py` uses Gymnasium's own checker to verify the environment complies with the standard API:

```python
from gymnasium.utils.env_checker import check_env

def test_check_env():
    env = make_env()
    check_env(env)
```

---

### Version 2 — Wrappers (`festivalMANAGER (wrappers)`)

This version adds **three Gymnasium wrappers** on top of `FestivalEnv` in `agent/train.py`. The environment files are identical to Version 1.

```python
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, NormalizeObservation

def _make_env(mode):
    env = FestivalEnv(mode=mode)
    env = TimeLimit(env, max_episode_steps=12)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    return env
```

**`TimeLimit`** automatically ends the episode after 12 steps by setting `truncated=True`.

**`RecordEpisodeStatistics`** tracks total reward, episode length, and wall time automatically, injecting them into `info["episode"]` at episode end. `_run_episode()` now returns `ep_stats` and the training loop reads `ep_stats.get("r")` and `ep_stats.get("l")`:

```python
if done:
    ep_stats = info.get("episode", {})   # only populated at episode end
    break
return total_reward, ep_stats
```

**`NormalizeObservation`** standardises observations to approximately mean 0 and variance 1. Because the observation range shifts from `[0, 1]` to roughly `[-3, 3]`, `_discretize()` is updated to use `np.digitize` with explicit bins:

```python
def _discretize(obs):
    bins = np.digitize(obs, bins=[-2, -1, 0, 1, 2, 3])
    result = 0
    for b in bins:
        result = result * 10 + int(b)
    return result
```

| Wrapper | What it adds | Why it is useful here |
|---------|-------------|----------------------|
| `TimeLimit` | Cuts episodes at 12 steps | Enforces the 12-slot day without touching `festival_env.py` |
| `RecordEpisodeStatistics` | Auto-logs reward, length, time into `info["episode"]` | Removes manual reward accumulation from the training loop |
| `NormalizeObservation` | Standardises observations to ~mean 0, variance 1 | Ensures all features are on the same scale for discretisation |

---

### Version 3 — Custom environment (`festivalMANAGER (versión entorno propio)`)

This version **removes the Gymnasium dependency from the environment** and replaces its built-in space classes with hand-written equivalents. It starts from Version 1 (not Version 2), because the three wrappers introduced in Version 2 do not make sense for our custom environment:

- **`TimeLimit`** is redundant — `FestivalEnv` already controls episode termination internally via `current_slot >= n_slots`, so adding the wrapper on top could cause conflicts if both count steps simultaneously.
- **`NormalizeObservation`** is counterproductive — `_build_obs()` already returns everything in `[0, 1]` using `np.clip`. Normalising data that is already normalised has no benefit and can distort values.
- **`RecordEpisodeStatistics`** is the only one that added something useful, but in the wrapper-free version we simply use `ep_reward` directly in the training loop, which is cleaner and more explicit.

The goal of this version is to demonstrate that we understand what Gymnasium's spaces actually do under the hood by reimplementing them from scratch.

#### Changes to `env/festival_env.py`

**Step 1:** Remove the Gymnasium imports:

```python
# REMOVED:
import gymnasium as gym
from gymnasium import spaces
```

**Step 2:** Add two custom classes right after the imports, replacing `spaces.MultiDiscrete` and `spaces.Box`:

```python
# Replaces gymnasium.spaces.MultiDiscrete
class MultiDiscreteSpace:
    def __init__(self, nvec, seed=None):
        self.nvec = np.array(nvec, dtype=np.int64)
        self.shape = (len(self.nvec),)
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return self._rng.integers(0, self.nvec, dtype=np.int64)

    def contains(self, action):
        action = np.asarray(action)
        if action.shape != self.shape:
            return False
        return all(0 <= action[i] < self.nvec[i] for i in range(len(self.nvec)))


# Replaces gymnasium.spaces.Box
class BoxSpace:
    def __init__(self, low, high, shape, dtype=np.float32, seed=None):
        self.low = float(low)
        self.high = float(high)
        self.shape = tuple(shape)
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return self._rng.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    def contains(self, obs):
        obs = np.asarray(obs)
        return (obs.shape == self.shape
                and bool(np.all(obs >= self.low))
                and bool(np.all(obs <= self.high)))
```

`FestivalEnv` now inherits from plain `object` instead of `gym.Env`, and uses the custom spaces:

```python
class FestivalEnv():
    ...
    self.action_space = MultiDiscreteSpace([n_artists + 1, n_stages, n_slots, 4])
    self.observation_space = BoxSpace(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
```

The `reset()` method also simplifies: the `super().reset(seed=seed)` call is removed since there is no longer a parent Gymnasium class.

#### Changes to `agent/train.py`

The action space check switches from Gymnasium's class to our own:

```python
# BEFORE (Version 1):
from gymnasium import spaces
if isinstance(env.action_space, spaces.MultiDiscrete):
    n_actions = int(np.prod(env.action_space.nvec))

# AFTER (Version 3):
from env.festival_env import MultiDiscreteSpace
if isinstance(env.action_space, MultiDiscreteSpace):
    n_actions = int(np.prod(env.action_space.nvec))
```

When switching to the custom environment, **the agent stopped learning entirely**. This was visible in the first chart produced by `visualizar.py` (training curve), which showed a flat line instead of the expected upward trend. The problem was in `_discretize()`: with 47 continuous variables in `[0, 1]`, the old formula (`× 10`) produced up to 10^47 unique state keys — a number so large the agent could never revisit any state and therefore could never update a Q-value meaningfully.

The fix was to select only the **4 most relevant variables** — happiness, revenue, time slot, and average queue length — and divide each into 5 buckets. This caps the state space at 5^4 = 625 possible states, which is small enough for the agent to revisit and learn from:

```python
# BEFORE (Versions 1 and 2):
def _discretize(obs):
    bins = np.clip((obs * 10).astype(int), 0, 9)
    result = 0
    for b in bins:
        result = result * 10 + int(b)
    return result

# AFTER (Version 3):
def _discretize(obs):
    happiness = obs[-3]
    revenue   = obs[-2]
    time_slot = obs[-1]
    avg_queue = np.mean(obs[44:49])

    h = int(np.clip(happiness * 5, 0, 4))
    r = int(np.clip(revenue   * 5, 0, 4))
    t = int(np.clip(time_slot * 5, 0, 4))
    q = int(np.clip(avg_queue * 5, 0, 4))

    return h * 125 + r * 25 + t * 5 + q
```

The wrappers are gone — `_make_env()` returns the bare `FestivalEnv` directly, and `_run_episode()` returns only `total_reward`.

#### Changes to `tests/test_env.py`

Since our environment no longer inherits from `gym.Env`, Gymnasium's `check_env` cannot be used. Test 5 (`test_check_env`) is replaced by `test_custom_spaces`, which verifies the behaviour of `MultiDiscreteSpace` and `BoxSpace` directly:

```python
# BEFORE (Version 1):
from gymnasium.utils.env_checker import check_env

def test_check_env():
    env = make_env()
    check_env(env)

# AFTER (Version 3):
from env.festival_env import FestivalEnv, MultiDiscreteSpace, BoxSpace

def test_custom_spaces():
    env = make_env()
    # MultiDiscreteSpace: sample() always produces valid actions
    for _ in range(20):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
    # contains() rejects wrong shapes and out-of-range values
    assert not env.action_space.contains(np.array([0, 0]))
    assert not env.action_space.contains(np.array([999, 999, 999, 999], dtype=np.int64))
    # BoxSpace: sample() stays within [0, 1] and correct shape
    for _ in range(10):
        obs_sample = env.observation_space.sample()
        assert obs_sample.min() >= 0.0 and obs_sample.max() <= 1.0
    # contains() accepts a real observation and rejects wrong shapes
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    assert not env.observation_space.contains(np.zeros(3))
```

---

### Version 4 — Search algorithms (`festivalMANAGER (versión entorno propio + implementación algoritmo)`)

This is the final version. It keeps everything from Version 3 unchanged and adds a new module, `navigation.py`, which applies **BFS (Breadth-First Search)** and **DFS (Depth-First Search)** — two classic graph search algorithms from the course — to a new problem inside the project: helping a festival attendee find their way from one zone to another.

The project structure gains two new files at the root:

```
├── navigation.py       ← BFS and DFS implementation + interactive navigator
└── festival_map.png    ← Visual map of the festival graph
```

No existing file is modified.

#### The festival as a graph

The festival grounds are modelled as an **undirected graph** where nodes are venue areas and edges are the passable paths between adjacent zones. There are 14 nodes:

```
         Baños Norte ── Barras Norte ─────────── Esc. Cúpula
              │               │    \                   │
         Esc. MAIN ──────────────── Zona Principal ── P. Comida
              │          Zona VIP        │    \            │
         Barras VIP                   Entrada   B. Principal  Z. Camping
                                         │          │    /
                                      Baños Sur  Esc. Césped
                                          \         │
                                           ── Barras Sur ──┘

Leyenda:  ○ Escenario   ○ Barras   ○ Zona   ○ Baños   ○ Entrada
```

| Category | Zones |
|----------|-------|
| Entrance | Entrada |
| Zones | Zona Principal, Zona VIP, Zona Camping |
| Stages | Escenario Main, Escenario Cúpula, Escenario Césped |
| Bars | Barras Norte, Barras Sur, Barras VIP, Barras Principal |
| Services | Puesto de Comida, Baños Norte, Baños Sur |

#### BFS — Breadth-First Search (shortest path guaranteed)

Uses a FIFO queue (`collections.deque`). Expands all nodes at depth *d* before moving to depth *d+1*, so the first time it reaches the destination it is guaranteed to be via the **fewest zone changes**. This is the optimal algorithm for finding the shortest path in an unweighted graph.

```python
from collections import deque

def bfs(graph, origin, destination):
    queue = deque([(origin, [origin])])
    visited = {origin}
    while queue:
        current, path = queue.popleft()   # FIFO — guarantees shortest path
        for neighbour in graph.get(current, []):
            if neighbour not in visited:
                new_path = path + [neighbour]
                if neighbour == destination:
                    return new_path
                visited.add(neighbour)
                queue.append((neighbour, new_path))
    return None
```

#### DFS — Depth-First Search (valid path, not guaranteed shortest)

Uses a LIFO stack. Goes as deep as possible before backtracking. Finds *a* valid path, but **does not guarantee** it is the shortest. The comparison between BFS and DFS results illustrates exactly why the choice of data structure (queue vs stack) changes the nature of the search.

```python
def dfs(graph, origin, destination):
    stack = [(origin, [origin])]
    visited = {origin}
    while stack:
        current, path = stack.pop()       # LIFO — explores depth first
        for neighbour in graph.get(current, []):
            if neighbour not in visited:
                new_path = path + [neighbour]
                if neighbour == destination:
                    return new_path
                visited.add(neighbour)
                stack.append((neighbour, new_path))
    return None
```

Both functions store the **full path** at each step (not just visited markers), so the complete route can be printed zone by zone. After running both algorithms, the module prints a side-by-side comparison: the full route for each, the number of steps, and a verdict. When the image `festival_map.png` is found it opens automatically; if the file is missing, the code continues without error so training is never blocked.

#### Integration in `main.py`

`navigation.py` runs as **Step 0**, before Q-Learning training starts. The user can query as many routes as they want, then continue:

```python
# Step 0 — BFS vs DFS navigator
from navigation import run_navigation
run_navigation()

# Step 1 — Q-Learning training
from agent.train import train_all_modes
train_all_modes()
```

---

