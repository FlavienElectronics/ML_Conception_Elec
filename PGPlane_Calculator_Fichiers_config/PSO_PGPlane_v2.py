import numpy as np
import subprocess
import time
import os
import threading
import queue
from sko.PSO import PSO
from sko.tools import set_run_mode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# =============================================================================
# 0. CONFIGURATION & DOSSIERS
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

BASE_SAVE_DIR = "Optimization_Runs"
def get_unique_dir(base_name):
    counter = 1
    while os.path.exists(f"{base_name}_{counter}"):
        counter += 1
    new_dir = f"{base_name}_{counter}"
    os.makedirs(new_dir)
    return new_dir

SAVE_PATH = get_unique_dir(BASE_SAVE_DIR)
print(f"{Colors.HEADER}>>> Mode Marge de Sécurité 20% Activé. Sauvegarde dans : {SAVE_PATH}{Colors.RESET}")

EXE_PATH = r'PGPlane_Calculator.exe' 
CONFIG_FILE = r'CasTest_8decap.pgconf'
DATA_FILE = 'port_file.txt'
OUT_FILE = 'out.txt'
PORT_NAMES_STR = "PC1;PC2;PC3;PC4;PC5;PC6;PC7;PC8"

CAP_SIZES = np.array([[3.2, 1.6], [2.0, 1.2], [2.0, 1.2], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [0.6, 0.3], [0.6, 0.3]])
CAP_RADII = np.array([np.sqrt(w**2 + h**2) / 2.0 for w, h in CAP_SIZES])

# --- PARAMÈTRES OPTIMISATION ---
NB_CAPS = 8
NB_VAR_OPT = NB_CAPS * 2 
X_CENTER, Y_CENTER = 20.0, 10.0
AREA_SIZE = 23.0
X_MIN, X_MAX = X_CENTER - AREA_SIZE/2, X_CENTER + AREA_SIZE/2
Y_MIN, Y_MAX = Y_CENTER - AREA_SIZE/2, Y_CENTER + AREA_SIZE/2

LOWER_BOUNDS, UPPER_BOUNDS = [], []
for i in range(NB_CAPS):
    r = CAP_RADII[i]
    LOWER_BOUNDS.extend([X_MIN + r, Y_MIN + r])
    UPPER_BOUNDS.extend([X_MAX - r, Y_MAX - r])

# Fréquences (91 pts)
FREQ_MIN, FREQ_MAX, PTS_PER_DEC = 1e6, 1e9, 30
NB_FREQ_PTS = int((np.log10(FREQ_MAX) - np.log10(FREQ_MIN)) * PTS_PER_DEC) + 1
FREQ_VECTOR = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NB_FREQ_PTS)

# --- TARGET AVEC MARGE DE 20% ---
Z_SAFETY_FACTOR = 0.8  # 100% - 20% = 0.8
Z_TARGET_NOMINAL = np.zeros(NB_FREQ_PTS)
F_CORNER, Z_FLAT = 100e6, 0.1
for i, f in enumerate(FREQ_VECTOR):
    Z_TARGET_NOMINAL[i] = Z_FLAT if f <= F_CORNER else Z_FLAT * (f / F_CORNER)

# La cible pour l'optimiseur est 20% plus basse
Z_TARGET_MATH = Z_TARGET_NOMINAL * Z_SAFETY_FACTOR

NB_POP_SIZE = 60
MAX_ITER_PER_BLOCK = 50 # On travaille par blocs d'itérations

# =============================================================================
# 1. LOGIQUE DE COÛT (CONTRAINTE FORTE)
# =============================================================================

def calculate_cost(z_matrix, population_matrix):
    global global_iter_count
    # 1. Coût Impédance avec Marge de 20%
    # On compare à Z_TARGET_MATH (80% de la cible nominale)
    z_target_col = Z_TARGET_MATH.reshape(-1, 1)
    
    # Violation de la marge : différence positive entre Z_simu et Z_cible_math
    violation = np.maximum(0, z_matrix - z_target_col)
    
    # On utilise un exposant fort pour que toute pointe au-dessus de la marge soit "catastrophique"
    # Pondération par fréquence pour stabiliser la HF
    weights = (FREQ_VECTOR / FREQ_MIN).reshape(-1, 1)
    cost_impedance = np.sum((violation**2) * weights, axis=0) * 1000

    # 2. Coût Collision (Cercle à Cercle)
    nb_individuals = population_matrix.shape[0]
    collision_penalties = np.zeros(nb_individuals)
    for n in range(nb_individuals):
        coords = population_matrix[n, :].reshape(NB_CAPS, 2)
        penalty_n = 0
        for i in range(NB_CAPS):
            for j in range(i + 1, NB_CAPS):
                dist = np.linalg.norm(coords[i] - coords[j])
                min_dist = CAP_RADII[i] + CAP_RADII[j]
                if dist < min_dist:
                    penalty_n += (min_dist - dist) * 100 # Pénalité linéaire
        collision_penalties[n] = penalty_n
        
    return cost_impedance + collision_penalties

# =============================================================================
# 2. LOGIQUE D'EXPLORATION CONTINUE
# =============================================================================

global_best_profile = None
condition_satisfied = False

def obj_function(p):
    global global_iter_count, global_best_so_far, global_best_profile, condition_satisfied
    
    z_matrix = launch_simu_engine(p)
    if z_matrix is None: return np.ones(p.shape[0]) * 1e9
    
    costs = calculate_cost(z_matrix, p)
    best_idx = np.argmin(costs)
    
    current_best_z = z_matrix[:, best_idx]
    
    # VERIFICATION DE LA CONDITION : Est-ce que tout le tracer est sous Z_TARGET_MATH ?
    # On vérifie si le maximum de la différence est <= 0
    margin_violation = np.max(current_best_z - Z_TARGET_MATH)
    
    global_iter_count += 1
    if costs[best_idx] < global_best_so_far:
        global_best_so_far = costs[best_idx]
        global_best_profile = current_best_z
        
        # Si la violation est nulle, on a gagné la marge de 20% !
        if margin_violation <= 0:
            condition_satisfied = True

    gui_queue.put({
        "iter": global_iter_count, "curr": costs[best_idx], "gbest": global_best_so_far,
        "pos": p[best_idx, :], "z_profile": current_best_z, "satisfied": condition_satisfied
    })
    
    status = f"{Colors.GREEN}CONFORME (-20%){Colors.RESET}" if margin_violation <= 0 else f"{Colors.RED}HORS-MARGE{Colors.RESET}"
    print(f" > Iter: {global_iter_count} | Cost: {global_best_so_far:.8f} | Statut: {status}")
    
    return costs

def launch_simu_engine(p):
    with open(DATA_FILE, 'w') as f:
        f.write(PORT_NAMES_STR + "\n")
        for i in range(p.shape[0]):
            line = ";".join([f"{val/1000.0:.6e}" for val in p[i, :]])
            f.write(line + "\n")
    args = [EXE_PATH, str(p.shape[0]), CONFIG_FILE, DATA_FILE, OUT_FILE]
    try:
        subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return np.loadtxt(OUT_FILE) if os.path.exists(OUT_FILE) else None
    except: return None

def run_pso_thread():
    set_run_mode(obj_function, 'vectorization')
    
    # Boucle d'exploration continue
    while not condition_satisfied:
        print(f"{Colors.BLUE}>>> Lancement d'un bloc d'exploration...{Colors.RESET}")
        pso = PSO(func=obj_function, n_dim=NB_VAR_OPT, pop=NB_POP_SIZE, max_iter=MAX_ITER_PER_BLOCK, 
                  lb=LOWER_BOUNDS, ub=UPPER_BOUNDS, w=0.7, c1=1.5, c2=1.5)
        pso.run()
        
        if condition_satisfied:
            print(f"{Colors.GREEN}>>> OPTIMISATION TERMINÉE : Marge de 20% atteinte !{Colors.RESET}")
            break
        else:
            print(f"{Colors.RED}>>> Marge non atteinte. Relance de l'exploration...{Colors.RESET}")

# =============================================================================
# 3. GUI
# =============================================================================

gui_queue = queue.Queue()
global_iter_count, global_best_so_far = 0, float('inf')
history_iter, history_current, history_gbest = [], [], []

def update_gui(frame):
    data = None
    while not gui_queue.empty(): data = gui_queue.get_nowait()
    if data:
        history_iter.append(data["iter"])
        history_current.append(data["curr"])
        history_gbest.append(data["gbest"])
        
        line_gbest.set_data(history_iter, history_gbest)
        line_curr.set_data(history_iter, history_current)
        ax_conv.relim(); ax_conv.autoscale_view()
        ax_cost.relim(); ax_cost.autoscale_view()
        
        # Affichage Z
        line_z_curr.set_data(FREQ_VECTOR, data["z_profile"])
        ax_z.relim(); ax_z.autoscale_view()

        # Nettoyage Map
        for patch in list(ax_map.patches):
            if patch.get_label() == 'cap_item': patch.remove()
        for line in list(ax_map.lines):
            if line.get_label() == 'cap_item': line.remove()
        
        coords = data["pos"].reshape((NB_CAPS, 2))
        color_status = 'green' if data["satisfied"] else 'blue'
        for i in range(NB_CAPS):
            circle = plt.Circle((coords[i,0], coords[i,1]), CAP_RADII[i], 
                                color=color_status, alpha=0.3, ec=color_status, lw=1, label='cap_item', zorder=10)
            ax_map.add_patch(circle)
            ax_map.plot(coords[i,0], coords[i,1], '.', color=color_status, ms=2, label='cap_item', zorder=11)
            annot_texts[i].set_position((coords[i,0], coords[i,1]))

        ax_map.set_title(f"Exploration Continue (Iter {data['iter']})")
        fig.savefig(os.path.join(SAVE_PATH, f"iteration_{data['iter']:03d}.png"))
    return line_gbest, line_curr

if __name__ == "__main__":
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1], figure=fig)
    ax_conv, ax_cost, ax_z, ax_map = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])

    # Tracers
    ax_z.loglog(FREQ_VECTOR, Z_TARGET_NOMINAL, 'r--', alpha=0.5, label='Target Nominale')
    ax_z.loglog(FREQ_VECTOR, Z_TARGET_MATH, 'g-', lw=2, label='Target -20% (Safe)')
    line_z_curr, = ax_z.loglog(FREQ_VECTOR, np.zeros(NB_FREQ_PTS), 'b-', label='Simulation')
    ax_z.set_title("Impédance PDN (Ohm)"); ax_z.grid(True, which="both", alpha=0.3); ax_z.legend()

    line_gbest, = ax_conv.plot([], [], 'b-o', lw=2, label='Global Best')
    line_curr, = ax_cost.plot([], [], 'r--s', ms=4, label='Iter Best')
    
    ax_map.set_xlim(-75, 75); ax_map.set_ylim(-40, 40)
    ax_map.add_patch(plt.Rectangle((-75, -40), 150, 80, fill=None, ec='k', lw=2))
    ax_map.add_patch(plt.Rectangle((X_MIN, Y_MIN), AREA_SIZE, AREA_SIZE, fill=None, ec='red', ls='--'))
    ax_map.plot(20, 10, 'rx', ms=8, label='IC')
    annot_texts = [ax_map.text(0, 0, f'C{i+1}', ha='center', va='center', fontsize=7, fontweight='bold', zorder=12) for i in range(NB_CAPS)]
    ax_map.set_aspect('equal')

    threading.Thread(target=run_pso_thread, daemon=True).start()
    ani = FuncAnimation(fig, update_gui, interval=500, blit=False, cache_frame_data=False)
    plt.show()