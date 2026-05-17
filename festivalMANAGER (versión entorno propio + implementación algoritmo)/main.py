"""
Main entry point of the Festival Manager project.
It executes training and evaluation, and generates figures to compare and analyze the results.

The workflow of main.py is:
    1. Trains Q-Learning agents in different modes (A, B, C)
    2. Evaluates the trained models in the simulated environment
    3. Generates figures and visualizations for the final report
    4. Prints a final summary with links to results and models
"""

# Import system tools
import os
import sys
 
# Obtain the folder where this script is located so that the imports work correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If BASE_DIR is not in sys.path, it adds it 
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
 
# STEP 0: BFS vs. DFS
print("=======================================================")
print(" PASO 0 — Navegación por el festival (BFS vs DFS)")
print("=======================================================")

# Import and run the interactive navigation module.
# The user can search as many routes as they want before training starts.
from navigation import run_navigation
run_navigation()



# STEP 1 — Q-Learning Training
print("=======================================================")
print(" PASO 1 — Entrenamiento Q-Learning (A, B y C)")
print("=======================================================")

# Import the training function for the three modes
from agent.train import train_all_modes
train_all_modes()
 

# STEP 2 — Evaluation of the trained models
print("=======================================================")
print(" PASO 2 — Evaluación de agentes entrenados")
print("=======================================================")

# Import the evaluation functions for the trained models and the random baseline
from evaluation.evaluate import evaluar_todos, baseline_aleatoria
baseline_aleatoria(n_episodios=30)
evaluar_todos(n_episodios=30)


# STEP 3 — Visualizations
print("=======================================================")
print(" PASO 3 — Generación de figuras y visualizaciones")
print("=======================================================")

# Import the functions used to generate the analysis and comparison charts
from evaluation.visualizar import (
    fig_curvas_entrenamiento,
    fig_ql_vs_aleatorio,
    fig_comparativa_barras,
    fig_scatter_revenue_happiness,
    fig_heatmap_horario,
)

# Run to generate the figures
fig_curvas_entrenamiento()
fig_ql_vs_aleatorio()
fig_comparativa_barras()
fig_scatter_revenue_happiness()

# The hourly heatmap is generated for each mode (A, B, C)
for modo in ["A", "B", "C"]:
    fig_heatmap_horario(modo=modo)
 

# Final resume
print("=======================================================")
print(" Resumen final:")
print("=======================================================")
print(" Modelos guardados en:   models/")
print(" Resultados en:          results/")
print(" Figuras en:             results/figuras/")
print(" Los archivos .html se abren en el navegador.")
print("=======================================================")