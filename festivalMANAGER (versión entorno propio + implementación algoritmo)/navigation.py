"""
In the navigation.py file, we implemented some of the techniques covered in class,
specifically the BFS (Breadth-First Search) and DFS (Depth-First Search) algorithms.

We chose these algorithms because we believe they are the most optimal and make the most sense
for our project, as they allow the user to enter their current location and receive
directions on how to reach their destination.

To do this, we modeled the festival as a graph, where:
    - Nodes represent the different areas (stages, bars, restrooms, entrance, etc.)
    - Edges represent the passable paths between adjacent areas

Additionally, we have implemented both algorithms so we can compare them:
    - BFS (Breadth-First Search): guarantees finding the shortest path
      (i.e., the one that passes through the fewest zones)
    - DFS (Depth-First Search): finds a valid path between the origin
      and destination, but does not guarantee that it is the shortest

Note: The DFS algorithm has been adapted to store full paths instead of
just traversal order, making it more useful for route finding in this context.

Then, in step 0 of the main function, we call the entire module so that it appears automatically before 
the visualizations and tables. 
"""

# Imports
from collections import deque  # BFS uses a queue (FIFO)
import os
from PIL import Image

#  FESTIVAL GRAPH
FESTIVAL_GRAPH = {
    "Entrada": [
        "Zona_Principal",
        "Barras_Principal",
        "Banos_Sur",
    ],
    "Zona_Principal": [
        "Entrada",
        "Escenario_Main",
        "Escenario_Cupula",
        "Barras_Norte",
        "Zona_VIP",
    ],
    "Escenario_Main": [
        "Zona_Principal",
        "Zona_VIP",
        "Barras_Norte",
        "Banos_Norte",
    ],
    "Escenario_Cupula": [
        "Zona_Principal",
        "Barras_Norte",
        "Puesto_Comida",
    ],
    "Escenario_Cesped": [
        "Zona_Camping",
        "Barras_Sur",
        "Banos_Sur",
    ],
    "Zona_VIP": [
        "Zona_Principal",
        "Escenario_Main",
        "Barras_VIP",
    ],
    "Zona_Camping": [
        "Escenario_Cesped",
        "Puesto_Comida",
        "Barras_Sur",
    ],
    "Barras_Norte": [
        "Zona_Principal",
        "Escenario_Main",
        "Escenario_Cupula",
        "Banos_Norte",
    ],
    "Barras_Sur": [
        "Escenario_Cesped",
        "Zona_Camping",
        "Banos_Sur",
    ],
    "Barras_VIP": [
        "Zona_VIP",
        "Escenario_Main",
    ],
    "Barras_Principal": [
        "Entrada",
        "Zona_Principal",
        "Puesto_Comida",
    ],
    "Puesto_Comida": [
        "Barras_Principal",
        "Escenario_Cupula",
        "Zona_Camping",
    ],
    "Banos_Norte": [
        "Barras_Norte",
        "Escenario_Main",
    ],
    "Banos_Sur": [
        "Entrada",
        "Escenario_Cesped",
        "Barras_Sur",
    ],
}

# For visualization
GROUPS = {
    "ENTRADA": ["Entrada"],
    "ZONAS": ["Zona_Principal", "Zona_VIP", "Zona_Camping"],
    "ESCENARIOS": ["Escenario_Main", "Escenario_Cupula", "Escenario_Cesped"],
    "BARRAS": ["Barras_Norte", "Barras_Sur", "Barras_VIP", "Barras_Principal"],
    "SERVICIOS": ["Puesto_Comida", "Banos_Norte", "Banos_Sur"],
}

ZONE_LABELS = {
    "Entrada": "Entrada principal",
    "Zona_Principal": "Zona Principal",
    "Escenario_Main": "Escenario MAIN",
    "Escenario_Cupula": "Escenario Cúpula",
    "Escenario_Cesped": "Escenario Césped",
    "Zona_VIP": "Zona VIP",
    "Zona_Camping": "Zona Camping",
    "Barras_Norte": "Barras Norte",
    "Barras_Sur": "Barras Sur",
    "Barras_VIP": "Barras VIP",
    "Barras_Principal": "Barras Principal",
    "Puesto_Comida": "Puesto de Comida",
    "Banos_Norte": "Baños Norte",
    "Banos_Sur": "Baños Sur",
}


#  BFS
# uses a queue (FIFO) to explore nodes level by level,
# ensuring the first time we reach the destination is via the shortest path
def bfs(graph, origin, destination):
    if origin == destination:
        return [origin]

    queue = deque()
    # store node + path taken so far
    queue.append((origin, [origin]))

    visited = set()
    visited.add(origin)

    while queue:
        # FIFO → guarantees shortest path
        current, path = queue.popleft() 

        for neighbour in graph.get(current, []):
            if neighbour in visited:
                continue
            new_path = path + [neighbour]
            if neighbour == destination:
                return new_path          
            visited.add(neighbour)
            queue.append((neighbour, new_path)) 

    return None  # no path exists


#  DFS
# uses a stack (LIFO) to explore as deep as possible before backtracking.
# In this implementation, we store the full path to allow route reconstruction.
def dfs(graph, origin, destination):
    if origin == destination:
        return [origin]
    
    # store node + full path 
    stack = [(origin, [origin])]

    visited = set()
    visited.add(origin)

    while stack:
        # LIFO → explores depth first
        current, path = stack.pop()  

        for neighbour in graph.get(current, []):
            if neighbour in visited:
                continue
            new_path = path + [neighbour]
            if neighbour == destination:
                return new_path         
            visited.add(neighbour)
            stack.append((neighbour, new_path))

    return None  # no path exists



#  DISPLAY HELPERS
def format_path(path):
    if path is None:
        return "  (no hay ruta disponible)"

    lines = []
    for i, node in enumerate(path):
        label = ZONE_LABELS.get(node, node)
        if i == 0:
            prefix = "  Salida  "
        elif i == len(path) - 1:
            prefix = "  Llegada "
        else:
            prefix = f"  Paso {i}   " if i < 10 else f"  Paso {i}  "
        lines.append(f"{prefix}->  {label}")

    # Summary of intermediate zones (everything between origin and destination)
    if len(path) > 2:
        intermedias = [ZONE_LABELS.get(n, n) for n in path[1:-1]]
        lines.append(f"\n  Zonas intermedias: {' -> '.join(intermedias)}")
    else:
        lines.append("\n  (zonas adyacentes: sin pasos intermedios)")

    lines.append(f"  Pasos totales: {len(path) - 1}")
    return "\n".join(lines)


def print_comparison(origin, destination, path_bfs, path_dfs):
    """Print a clear step-by-step comparison of BFS and DFS results."""
    print("\n" + "=" * 60)
    print(f"  RUTA SOLICITADA:")
    print(f"  Desde : {ZONE_LABELS.get(origin, origin)}")
    print(f"  Hasta : {ZONE_LABELS.get(destination, destination)}")
    print("=" * 60)

    # BFS result
    print("\n  BFS — Camino MAS CORTO (menos zonas que cruzar):")
    print("  " + "-" * 44)
    if path_bfs:
        print(format_path(path_bfs))
    else:
        print("  No se encontro ruta.")

    # DFS result
    print("\n  DFS — Un camino CUALQUIERA (puede ser mas largo):")
    print("  " + "-" * 44)
    if path_dfs:
        print(format_path(path_dfs))
    else:
        print("  No se encontro ruta.")

    # Comparison verdict
    print("\n" + "=" * 60)
    print("  COMPARACION BFS vs DFS:")
    if path_bfs and path_dfs:
        diff = len(path_dfs) - len(path_bfs)
        if diff == 0:
            print("  Ambos encontraron el mismo recorrido en este caso.")
        elif diff > 0:
            print(f"  BFS: {len(path_bfs) - 1} paso(s)  |  DFS: {len(path_dfs) - 1} paso(s)")
            print(f"  BFS es {diff} paso(s) mas corto que DFS.")
            print("  Esto demuestra por que BFS garantiza el camino minimo")
            print("  en grafos sin pesos, y DFS no lo garantiza.")
        else:
            print(f"  DFS fue mas corto aqui ({abs(diff)} paso(s) menos).")
            print("  Puede ocurrir por el orden del grafo, pero")
            print("  BFS SIEMPRE garantiza el minimo, DFS no.")
    print("=" * 60)


def show_map():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "festival_map.png")

    if not os.path.exists(image_path):
        print("No se encontró la imagen del mapa.")
        return

    img = Image.open(image_path)
    img.show()


#  USER INTERFACE
def ask_zone(prompt):
    """Ask the user to pick a valid zone by number grouped by categories."""

    print(f"\n  {prompt}")

    keys_ordered = []
    counter = 1

    for group_name, keys in GROUPS.items():
        print(f"\n  {group_name}:")
        for key in keys:
            print(f"    {counter:2d}. {ZONE_LABELS[key]}")
            keys_ordered.append(key)
            counter += 1

    while True:
        try:
            choice = int(input("\n  Elige un número: "))
            if 1 <= choice <= len(keys_ordered):
                return keys_ordered[choice - 1]
            print(f"  Número inválido. Elige entre 1 y {len(keys_ordered)}.")
        except ValueError:
            print(" Introduce un número entero.")



#  PUBLIC FUNCTION — called by main.py
def run_navigation():
    """
    Interactive navigation session: shows the festival map, lets the user
    search routes with BFS and DFS, and loops until they choose to stop.
    Called from main.py before the Q-Learning training begins.
    """
    print("\n" + "=" * 60)
    print("  FESTIVAL MANAGER — NAVEGADOR DE ZONAS")
    print("  Algoritmos: BFS vs DFS")
    print("=" * 60)

    show_map()

    while True:
        # Ask for origin and destination
        origin = ask_zone("¿Desde dónde partes?")
        destination = ask_zone("¿A dónde quieres ir?")

        if origin == destination:
            print("\n  Ya estás ahí, ¡no tienes que moverte!")
        else:
            # Run both algorithms
            path_bfs = bfs(FESTIVAL_GRAPH, origin, destination)
            path_dfs = dfs(FESTIVAL_GRAPH, origin, destination)

            # Show comparison
            print_comparison(origin, destination, path_bfs, path_dfs)

        # Ask if the user wants to search again
        print("\n  ¿Quieres buscar otra ruta? (s/n): ", end="")
        again = input().strip().lower()
        if again != "s":
            print("\n  ¡Disfruta el festival! \n")
            break


#  MAIN — direct execution
if __name__ == "__main__":
    run_navigation()