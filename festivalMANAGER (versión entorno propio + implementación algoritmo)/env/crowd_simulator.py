"""
Simulate how the crowd moves around the festival at each time slot. 

The main logic is:
    1. Each artist has a draw_power that pulls people toward the stage where they perform
    2. During breaks, people move to the bars and rest areas
    3. The occupancy of each zone changes step by step
    4. Bar queues depend on how many people want to buy and how many staff are working
    5. Happiness is calculated using the artist quality + how comfortable people feel 

"""

# Libraries
import numpy as np

#  These values define how our crowd is going to behave
BAR_PULL_BASE = 0.25   # percentage of people going to bars normally 
BAR_PULL_BREAK = 0.55   # percentage of people going to bars during breaks
COMFORT_PEAK = 0.50   # ocupancy for maximum enjoyment
COMFORT_THRESHOLD = 0.75   # over this limit people star feeling unconfortable and their happiness lowers
QUEUE_DISCOMFORT = 0.8    # how many people hate waiting in long lines
HAPPINESS_DECAY = 0.03   # drop of happiness if nothing exciting is happening


class CrowdSimulator:
    def __init__(self, config, seed=0, noise_std=0.02):
        self.config = config
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.n_stages = config["n_stages"]
        self.n_bars = config["n_bars"]
        self.n_zones = config["n_zones"]

        # Fast access to properties like stage capacity and how fast bars can serve people
        self.stage_caps = np.array(
            [s["capacity_norm"] for s in config["stages"]], dtype=float)
        self.bar_rates  = np.array(
            [b["service_rate"]  for b in config["bars"]],   dtype=float)

    def _noise(self, std=None):
        # Small random variation to make the movement look more natural and less like a math formula
        s = std if std is not None else self.noise_std
        return float(self.rng.normal(0, s)) if s > 0 else 0.0

    def _clip(self, v):
        # To keep values between 0.0 and 1.0
        return float(np.clip(v, 0.0, 1.0))

    def compute_stage_occupancy(self, schedule_row, prev_stage_occ, time_step):
        # Logic to move people in and out of the stages based on the current artists
        occ = prev_stage_occ.copy()

        for i in range(self.n_stages):
            artist_id = int(schedule_row[i])

            if artist_id == -1:
                # If no one is playing, people leave the stage area slowly
                decay = 0.20 + 0.10 * (1 - time_step)
                occ[i] = max(0.0, occ[i] - decay)
            else:
                # If an artist is playing, people are attracted by their popularity
                artist = self.config["artists"][artist_id]
                target = artist["draw_power"] * self.stage_caps[i]
                alpha  = 0.35   # speed at which the crowd moves toward the stage
                occ[i] = (1 - alpha) * occ[i] + alpha * target

            # Add a bit of noise to simulate random movement
            occ[i] = self._clip(occ[i] + self._noise())

        return occ

    def compute_bar_demand(self, schedule_row, stage_occ, time_step, staff_levels):
        # Calculate how many people want to buy drinks based on what's happening on stage
        max_draw = 0.0
        for i in range(self.n_stages):
            aid = int(schedule_row[i])
            if aid != -1:
                max_draw = max(max_draw, self.config["artists"][aid]["draw_power"])

        # If the artists are boring or it's a break, more people go to the bars
        bar_pull = BAR_PULL_BASE + (1 - max_draw) * (BAR_PULL_BREAK - BAR_PULL_BASE)

        # Time effect: demand changes during the night
        time_bonus = 0.15 * np.sin(np.pi * time_step)

        # People who aren't at a stage are considered "free" to go to bars
        audience_free = max(0.0, 1.0 - float(np.mean(stage_occ)))

        demand = np.empty(self.n_bars)
        for i, bar in enumerate(self.config["bars"]):
            zone_factor = 1.0 + 0.1 * (bar["zone"] % 3)
            demand[i]   = (bar_pull * audience_free + time_bonus) * zone_factor
            demand[i]   = self._clip(demand[i] + self._noise(0.03))

        # Queue length = people wanting to buy minus the staff's ability to serve them
        effective_rate = self.bar_rates * (1.0 + 0.5 * np.array(staff_levels) / 3.0)
        rate_norm      = effective_rate / (self.bar_rates.max() * 1.5 + 1e-9)
        queue_len      = np.clip(np.maximum(0.0, demand - rate_norm), 0.0, 1.0)

        return demand, queue_len

    def compute_zone_occupancy(self, stage_occ, bar_demand, prev_zone_occ):
        # Keeps track of how full different general areas are 
        zone_occ = prev_zone_occ.copy()

        # Stage areas
        for i in range(min(self.n_stages, self.n_zones)):
            zone_occ[i] = self._clip(stage_occ[i] + self._noise(0.01))

        # Bars and rest areas
        avg_bar = float(np.mean(bar_demand))
        for z in range(self.n_stages, self.n_zones):
            target    = 0.3 + 0.5 * avg_bar
            zone_occ[z] = self._clip(0.7 * zone_occ[z] + 0.3 * target + self._noise(0.01))

        return zone_occ

    def compute_happiness(self, schedule_row, stage_occ, queue_len, prev_happiness, time_step):
        # 1. Artistic quality: people are happier if they like the artist and the sound is good
        art_quality = 0.0
        n_active    = 0
        for i in range(self.n_stages):
            aid = int(schedule_row[i])
            if aid != -1:
                artist = self.config["artists"][aid]
                stage  = self.config["stages"][i]
                art_quality += artist["popularity"] * stage["sound_quality"] * stage_occ[i]
                n_active    += 1
        art_score = art_quality / max(n_active, 1) if n_active > 0 else 0.0

        # 2. Comfort: being in a crowd is fun, but being crushed is not
        comfort_scores = []
        for occ in stage_occ:
            if occ <= COMFORT_PEAK:
                comfort_scores.append(occ / COMFORT_PEAK)
            elif occ <= COMFORT_THRESHOLD:
                comfort_scores.append(1.0)
            else:
                excess = (occ - COMFORT_THRESHOLD) / (1 - COMFORT_THRESHOLD + 1e-9)
                comfort_scores.append(max(0.0, 1.0 - 2.0 * excess))
        comfort_score = float(np.mean(comfort_scores)) if comfort_scores else 0.5

        # 3. Bar penalty: long queues make people annoyed
        queue_penalty = float(np.mean(queue_len)) * QUEUE_DISCOMFORT

        # 4. Final score: combine everything with previous happiness (inertia) and a small decay
        instant = float(np.clip(0.45 * art_score + 0.35 * comfort_score - 0.20 * queue_penalty,
                                0.0, 1.0))
        happiness = float(np.clip(0.65 * prev_happiness + 0.35 * instant - HAPPINESS_DECAY,
                                  0.0, 1.0))
        return happiness


    def step(self, schedule_row, staff_levels, prev_stage_occ,
             prev_zone_occ, prev_happiness, time_step):
        # The main function that runs all calculations for a single time slot
        # It calculates occupancy, demand, and happiness, then returns the new state
        
        stage_occ = self.compute_stage_occupancy(schedule_row, prev_stage_occ, time_step)
        bar_demand, queue_len = self.compute_bar_demand(
            schedule_row, stage_occ, time_step, staff_levels)
        zone_occ  = self.compute_zone_occupancy(stage_occ, bar_demand, prev_zone_occ)
        happiness = self.compute_happiness(
            schedule_row, stage_occ, queue_len, prev_happiness, time_step)

        return {
            "stage_occ":  stage_occ,
            "zone_occ":   zone_occ,
            "bar_demand": bar_demand,
            "queue_len":  queue_len,
            "happiness":  happiness,
        }



# MAIN — manual execution 
if __name__ == "__main__":
    from data_generator import generate_festival_config

    cfg = generate_festival_config(n_stages=3, n_slots=12, n_artists=15, seed=42)
    sim = CrowdSimulator(cfg, seed=0, noise_std=0.0)

    stage_occ = np.zeros(cfg["n_stages"])
    zone_occ  = np.zeros(cfg["n_zones"])
    happiness = 0.6
    staff     = np.full(cfg["n_bars"], 2.0)

    print(f"{'Slot':>4} | {'StageOcc':20s} | {'Happiness':>9} | {'AvgQueue':>8}")
    print("-" * 52)
    for slot in range(6):
        # Slots 0-2: artista 0 en escenario 0; slots 3-5: descanso
        schedule = np.full(cfg["n_stages"], -1)
        if slot < 3:
            schedule[0] = 0
        result    = sim.step(schedule, staff, stage_occ, zone_occ, happiness, slot / 12)
        stage_occ = result["stage_occ"]
        zone_occ  = result["zone_occ"]
        happiness = result["happiness"]
        occ_str   = "  ".join(f"{o:.2f}" for o in stage_occ)
        print(f"  {slot:2d} | {occ_str:20s} | {happiness:9.3f} | {np.mean(result['queue_len']):8.3f}")