"""
To evaluate the project. It only performs measurements, since the models are already loaded in agent/train.py.

The workflow of evaluate.py is:
    1. Load trained models (Q-tables)
    2. Run evaluation episodes without exploration (epsilon=0)
    3. Calculate metrics (reward, happiness, revenue, safety) and save them to CSV
    4. Compare with a random baseline
    5. Generate a final summary for analysis
"""

# Libraries
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from agent.train import _discretize
 
# Define project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # root project
RESULTS_DIR  = os.path.join(BASE_DIR, "results") # folder to save the results
MODELS_DIR = os.path.join(BASE_DIR, "models") # folder for the trained Q-tables
# If BASE_DIR is not in sys.path, it adds it 
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# To ensure that the results folder exists before saving the CSV files
os.makedirs(RESULTS_DIR, exist_ok=True)
# Import the environment to be able to load the Q-tables and run the episodes
from env.festival_env import FestivalEnv
 

# Load the pre-trained Q-table 
def cargar_qtable(modo, models_dir=MODELS_DIR):
    ruta = os.path.join(models_dir, f"qtable_{modo}.npy")
    # If the file cannot be found, display an error message with a clear explanation
    # to avoid running the evaluation without trained models 
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró la Q-table en: {ruta}\n"
            f"Ejecuta primero: python agent/train.py --mode {modo}"
        )
 
    # Load the data from the Q-table, and use .item() to convert the loaded array into a Python dictionary
    datos = np.load(ruta, allow_pickle=True).item()


    env_tmp = FestivalEnv(mode=modo) # create a temporary environment to determine the number of actions
    n_acc = int(np.prod(env_tmp.action_space.nvec)) # total number of discrete actions (flattened)
    env_tmp.close() # close the temporary environment
 
    # Reconstruct the Q-table by loading the learned values, and use defaultdict so 
    # that any unseen state has a value of zero
    qtable = defaultdict(lambda: np.zeros(n_acc))
    # Iterating through the loaded dictionary (data) and copy each 
    # state along with its Q-vector into the new Q-table
    for k, v in datos.items():
        qtable[int(k)] = v

    # Print the evaluated mode (A, B, C) and the number of known states 
    # in its Q-table to the output screen (Step 2 of main.py )
    print(f"  Q-table modo {modo} cargada — {len(qtable)} estados conocidos")
    return qtable # Return the reconstructed Q-table as a defaultdict
 
 
# Greedy action: given a state, it returns the action with the highest Q-value. (without exploration --> epsilon=0)
def accion_greedy(qtable, estado, env):
    # Get the Q-values for the given state from the Q-table. If the state has never been seen during training,
    # it will return an array of zeros due to the use of defaultdict.
    valores = qtable[estado] 
    
    # If all values are zero, it means that the state was never seen during training
    if np.all(valores == 0):
        # So we choose a random action to avoid always selecting the first action (which would be the case if we used np.argmax on an array of zeros)
        # This ensures that we can still evaluate the agent's performance in states that were not encountered
        return env.action_space.sample()
 
    # Convert the flat index of the best action into a multi-dimensional action
    # REMEMBER : nvec = [artist,stage,slot,staff] (auto-generated attribute by spaces.MultiDiscrete, defined in festival_env.py)
    # For example: 0 --> [0,0,0,0], 1 --> [0,0,0,1], 2 --> [0,0,1,0], 3 --> [0,0,1,1], ...
    mejor_idx = int(np.argmax(valores))
    accion = np.zeros(len(env.action_space.nvec), dtype=np.int64)
    for i in range(len(env.action_space.nvec) - 1, -1, -1): # right to left: staff → artist
        accion[i] = mejor_idx % env.action_space.nvec[i] # action for the current dimension
        mejor_idx //= env.action_space.nvec[i] # integer division to go to the next dimension
    return accion
 

# Evaluate each mode individually
def evaluar_modo(modo, n_episodios=30):
    # Function that evaluates a trained Q-learning agent in a given mode (A, B or C)
    # By default, it runs 30 episodes to obtain statistically meaningful results
    
    nombres = {"A": "Máx. rentabilidad", 
               "B": "Máx. satisfacción", 
               "C": "Equilibrio"} # labels for the modes, easy understanding 
    
    print(f"\nEvaluando modo {modo} — {nombres[modo]} ({n_episodios} episodios)...") # This would be printed in Step 2 of main.py
 
    qtable = cargar_qtable(modo) # Load the pre-trained Q-table corresponding to the selected mode
    env = FestivalEnv(mode=modo, seed=42) # Create a new environment instance for the evaluation, with a fixed seed (42) for reproducibility
    filas = [] # create an empty list to store the results of each episode

    # Iterating for each episode to evaluate the trained agent
    for ep in range(n_episodios):
        obs, _  = env.reset() # reset the environment at the begining of each episode
        estado = _discretize(obs)  # Convert continuous observation into a discrete state. This is necessary because the Q-table  is tabular (discrete state)
        total_r  = 0.0 # initialize the total reward for the episode to zero, so it can accumulate the rewards obtained at each step of the episode
        h_pasos = [] # create an empty list to store the happiness values at each step of the episode (for computing the average hapiness)
        done = False # to indicate that the episode is not finished yet, and when it becomes TRUE (has finished), the loop will stop and the results will be stored
 
        # It enters into this loop if done is false, so it means that the episode is still ongoing. 
        while not done:
            accion = accion_greedy(qtable, estado, env) # Select the best action according to the learned Q-table (greedy policy)
            obs, r, term, trunc, info = env.step(accion) 
            done = term or trunc
            estado = _discretize(obs)
            total_r            += r
            h_pasos.append(info.get("happiness", 0.0))

        # Store results of the episode  
        filas.append({
            "episodio": ep + 1, # episode number (starting from 1)
            "recompensa_total": round(total_r, 4), # Total reward accumulated during the episode (rounded to 4 decimals)
            "happiness_media": round(float(np.mean(h_pasos)), 4), # Average happiness across all steps in the episode
            "revenue_total": round(info.get("revenue_so_far", 0.0), 4), # Total revenue accumulated during the episode
            "violaciones_seguridad": info.get("safety_violations", 0), # Number of safety constraint violations during the episode (p.e. overcrowding)
            "modo": modo, # to identify the mode
            "modo_etiqueta": nombres[modo], # label for the mode, easy understanding
        })
 
    env.close() # Close the temporary environment after evaluating the 30 episodes 
    return pd.DataFrame(filas) # Convert the list created and stored with the results into a DataFrame 
 
# Aleatory baseline: it runs episodes using a random policy (instead of the greedy policy based on the Q-table)
# the same structure as evaluar_modo, but instead of using the trained Q-table to select the action, it uses a random action (env.action_space.sample()) at each step of the episode.
def baseline_aleatoria(n_episodios=30, modo="C"):
    print(f"\nBaseline aleatoria ({n_episodios} episodios, modo {modo})...")
    env = FestivalEnv(mode=modo, seed=42) 
    filas = []
 
    for ep in range(n_episodios):
        obs, _ = env.reset()
        total_r = 0.0
        h_pasos = []
        done = False
 
        while not done:
            accion = env.action_space.sample() 
            obs, r, term, trunc, info = env.step(accion)
            done = term or trunc
            total_r += r
            h_pasos.append(info.get("happiness", 0.0))
 
        filas.append({
            "episodio": ep + 1,
            "recompensa_total": round(total_r, 4),
            "happiness_media": round(float(np.mean(h_pasos)), 4),
            "revenue_total": round(info.get("revenue_so_far", 0.0), 4),
            "violaciones_seguridad": info.get("safety_violations", 0),
            "modo": "Aleatoria",
            "modo_etiqueta": "Aleatoria (baseline)",
        })
 
    env.close()
    df = pd.DataFrame(filas)
    ruta = os.path.join(RESULTS_DIR, "baseline_aleatoria.csv")
    df.to_csv(ruta, index=False)
    print(f"  Guardado: {ruta}")
    return df
 

#  Evaluate the three modes and combine the results into a single CSV file 
def evaluar_todos(n_episodios=30):
    frames = []
    for modo in ["A", "B", "C"]:
        df = evaluar_modo(modo, n_episodios=n_episodios)
        frames.append(df)
 
    combinado = pd.concat(frames, ignore_index=True)
    ruta = os.path.join(RESULTS_DIR, "evaluacion_modos.csv")
    combinado.to_csv(ruta, index=False)
    print(f"\nEvaluación completa guardada: {ruta}")
    return combinado
 
 
# MAIN — manual execution to run the evaluation without executing main.py
if __name__ == "__main__":
    baseline_aleatoria(n_episodios=30)
    evaluar_todos(n_episodios=30)