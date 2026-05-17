"""
This script generates all the configuration data for the festival:
  - Artists (popularity, names, how long they play)
  - Stages (capacity, sound quality)
  - Bars (how fast they serve drinks)
  - Zones of the venue
"""

# Libraries
import numpy as np


#  List of possible names for artists, stages, zones, and bars to generate more realistic configs

ARTIST_NAMES = [
    "Hey Kid", "C. Tangana", "Rosalia", "Bad Bunny", "Justin Bieber",
    "Morat", "Izal", "Serko", "Kanye West", "Delaossa",
    "Guitarricadelafuente", "Quevedo", "Myke Towers", "Melendi", "Rels B",
    "Hard Gz", "Miranda", "Eminem", "Estopa", "Maldita Nerea",
]

STAGE_NAMES = [
    "Escenario MAIN",
    "Escenario Cupula",
    "Escenario Cesped",
]

ZONE_NAMES = [
    "Zona Principal",
    "Zona Barras Norte",
    "Zona Camping",
    "Zona VIP",
    "Zona Banos",
]

BAR_NAMES = [
    "Barra Norte",
    "Barra Sur",
    "Barra VIP",
    "Puesto Comida",
    "Barra Principal",
]


# Main generator function
def generate_festival_config(
    n_stages=3,
    n_slots=12,
    n_artists=15,
    n_bars=5,
    total_attendees=8000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # Artists
    artists = []
    for i in range(n_artists):
        # We want a mix: 2 big Headliners and the rest are smaller artists
        if i < 2:
            popularity = float(rng.uniform(0.85, 1.0)) # High popularity for headliners
        else:
            # Random math to get more mid-low tier artists than superstars
            u1 = float(rng.uniform(0, 1))
            u2 = float(rng.uniform(0, 1))
            popularity = round(min(u1, u2) * 1.5, 3)
            popularity = float(np.clip(popularity, 0.1, 0.84))

        # How much of the crowd this artist "pulls" to the stage
        draw_power = round(0.4 + 0.6 * popularity, 3)

        # Most artists play for 1 or 2 hours (slots)
        dur_options = [1, 2, 2, 3]
        set_duration = int(dur_options[rng.integers(len(dur_options))])

        artists.append({
            "id":               i,
            "name":             ARTIST_NAMES[i % len(ARTIST_NAMES)],
            "popularity":       round(popularity, 3),
            "set_duration_slots": set_duration,
            "draw_power":       draw_power,
        })

    # Stages
    base_cap = total_attendees // n_stages
    stages = []
    for i in range(n_stages):
        capacity = int(rng.integers(int(base_cap * 0.6), int(base_cap * 1.4)))
        stages.append({
            "id":            i,
            "name":          STAGE_NAMES[i % len(STAGE_NAMES)],
            "capacity":      capacity,
            "capacity_norm": round(capacity / total_attendees, 4),
            # Main stage has better sound
            "sound_quality": round(float(rng.uniform(0.6 + 0.1 * (n_stages - i - 1), 1.0)), 3),
            "location_zone": i % min(5, n_stages + 2),
        })

    # Zones
    n_zones = min(5, n_stages + 2)
    zones = []
    for i in range(n_zones):
        zones.append({
            "id":  i,
            "name": ZONE_NAMES[i % len(ZONE_NAMES)],
            "max_capacity": int(rng.integers(1000, total_attendees // 2)),
        })

    # Bars
    bars = []
    for i in range(n_bars):
        bars.append({
            "id":  i,
            "name": BAR_NAMES[i % len(BAR_NAMES)],
            "zone": i % n_zones,
            "service_rate": round(float(rng.uniform(10, 30)), 1),
            "max_queue": int(rng.integers(20, 60)),
        })

    return {
        "n_stages": n_stages,
        "n_slots": n_slots,
        "n_zones": n_zones,
        "n_bars": n_bars,
        "total_attendees":  total_attendees,
        "artists": artists,
        "stages": stages,
        "bars": bars,
        "zones": zones,
    }

# Helper functions to convert the dict data into numpy arrays for the simulator
def get_artist_popularity_vector(config):
    return np.array([a["popularity"] for a in config["artists"]])


def get_stage_capacity_vector(config):
    return np.array([s["capacity_norm"] for s in config["stages"]])


def get_bar_service_rates(config):
    return np.array([b["service_rate"] for b in config["bars"]])



# MAIN — manual execution to see the generated config
if __name__ == "__main__":
    config = generate_festival_config(n_stages=3, n_slots=12, n_artists=15)

    print("=== Festival Config ===")
    print(f"  Escenarios: {config['n_stages']}  |  Slots: {config['n_slots']}"
          f"  |  Aforo: {config['total_attendees']}")
    print(f"  Artistas  : {len(config['artists'])}")
    print(f"  Barras    : {len(config['bars'])}")
    print()

    headliners = [a for a in config["artists"] if a["popularity"] >= 0.85]
    print(f"Headliners ({len(headliners)}):")
    for a in headliners:
        print(f"  [{a['id']:2d}] {a['name']:20s}  pop={a['popularity']:.2f}"
              f"  draw={a['draw_power']:.2f}  dur={a['set_duration_slots']} slots")
    print()
    print("Escenarios:")
    for s in config["stages"]:
        print(f"  [{s['id']}] {s['name']:25s}  aforo={s['capacity']:5d}"
              f"  sonido={s['sound_quality']:.2f}")