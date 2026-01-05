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
# 0. UTILITAIRES & CONFIG
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXE_PATH = os.path.join(SCRIPT_DIR, 'PGPlane_Calculator.exe')
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'CasTest_8decap.pgconf')
DATA_FILE = os.path.join(SCRIPT_DIR, 'port_file.txt')
OUT_FILE = os.path.join(SCRIPT_DIR, 'out.txt')

PORT_NAMES_STR = "PC1;PC2;PC3;PC4;PC5;PC6;PC7;PC8"

# --- PARAMÈTRES ---
NB_CAPS = 8
NB_VAR_OPT = NB_CAPS * 2 
X_CENTER, Y_CENTER = 20.0, 10.0
AREA_SIZE = 23.0
HALF_SIZE = AREA_SIZE / 2.0
X_MIN, X_MAX = X_CENTER - HALF_SIZE, X_CENTER + HALF_SIZE
Y_MIN, Y_MAX = Y_CENTER - HALF_SIZE, Y_CENTER + HALF_SIZE

LOWER_BOUNDS = []
UPPER_BOUNDS = []
for _ in range(NB_CAPS):
    LOWER_BOUNDS.extend([X_MIN, Y_MIN])
    UPPER_BOUNDS.extend([X_MAX, Y_MAX])

FREQ_MIN, FREQ_MAX = 1e6, 1e9
PTS_PER_DEC = 30
NB_FREQ_PTS = int((np.log10(FREQ_MAX) - np.log10(FREQ_MIN)) * PTS_PER_DEC) + 1
FREQ_VECTOR = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NB_FREQ_PTS)

Z_TARGET_VECTOR = np.zeros(NB_FREQ_PTS)
F_CORNER, Z_FLAT = 100e6, 0.1
for i, f in enumerate(FREQ_VECTOR):
    if f <= F_CORNER: Z_TARGET_VECTOR[i] = Z_FLAT
    else: Z_TARGET_VECTOR[i] = Z_FLAT * (f / F_CORNER)

NB_POP_SIZE = 40
NB_MAX_ITER = 30
W_INERTIA = 0.8
C1_COGNITIVE = 0.5
C2_SOCIAL = 0.5

# =============================================================================
# 1. GESTION DES DONNÉES PARTAGÉES (THREAD-SAFE)
# =============================================================================

gui_queue = queue.Queue()

history_iter = []
history_current = []
history_gbest = []

global_iter_count = 0
global_best_so_far = float('inf')

# =============================================================================
# 2. MOTEUR DE CALCUL (Sera exécuté dans un Thread séparé)
# =============================================================================

def write_input_file(population_matrix):
    nb_individuals = population_matrix.shape[0]
    with open(DATA_FILE, 'w') as f:
        f.write(PORT_NAMES_STR + "\n")
        for i in range(nb_individuals):
            coords_meters = population_matrix[i, :] / 1000.0
            line_str = ";".join([f"{val:.6e}" for val in coords_meters])
            f.write(line_str + "\n")

def calculate_cost_from_z(z_matrix):
    z_target_col = Z_TARGET_VECTOR.reshape(-1, 1)
    diff = z_matrix - z_target_col
    penalty_matrix = np.maximum(0, diff)
    return np.sum(penalty_matrix, axis=0)

def launch_simu_engine(population_matrix):
    nb_individuals = population_matrix.shape[0]
    write_input_file(population_matrix)
    args = [EXE_PATH, str(nb_individuals), CONFIG_FILE, DATA_FILE, OUT_FILE]

    try:
        subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None

    if os.path.exists(OUT_FILE):
        try:
            return np.loadtxt(OUT_FILE)
        except:
            return None
    return None

def obj_function(p):
    global global_iter_count, global_best_so_far
    
    z_matrix = launch_simu_engine(p)
    
    if z_matrix is None:
        return np.ones(p.shape[0]) * 1e9
    
    costs = calculate_cost_from_z(z_matrix)
    
    current_min = np.min(costs)
    best_idx = np.argmin(costs)
    current_best_pos = p[best_idx, :]
    
    global_iter_count += 1
    if current_min < global_best_so_far:
        global_best_so_far = current_min

    data_packet = {
        "iter": global_iter_count,
        "curr": current_min,
        "gbest": global_best_so_far,
        "pos": current_best_pos
    }
    gui_queue.put(data_packet)
    
    color = Colors.GREEN if current_min <= global_best_so_far else Colors.RED
    print(f" > Itération : {global_iter_count}\t{Colors.BOLD}|{Colors.RESET} Cost :\t{color}{current_min:.4f}{Colors.RESET}")

    return costs

def run_pso_thread():
    print(" >>> Démarrage du Thread de calcul PSO...")
    set_run_mode(obj_function, 'vectorization')
    pso = PSO(func=obj_function, n_dim=NB_VAR_OPT, pop=NB_POP_SIZE, max_iter=NB_MAX_ITER, 
              lb=LOWER_BOUNDS, ub=UPPER_BOUNDS, w=W_INERTIA, c1=C1_COGNITIVE, c2=C2_SOCIAL)
    pso.run()
    print(" >>> Fin du calcul PSO.")

# =============================================================================
# 3. INTERFACE GRAPHIQUE (MAIN THREAD)
# =============================================================================

def update_gui(frame):
    updated = False
    last_data = None

    while not gui_queue.empty():
        try:
            last_data = gui_queue.get_nowait()
            history_iter.append(last_data["iter"])
            history_current.append(last_data["curr"])
            history_gbest.append(last_data["gbest"])
            updated = True
        except queue.Empty:
            break
    
    if updated and last_data:
        line_gbest.set_data(history_iter, history_gbest)
        ax_conv.relim()
        ax_conv.autoscale_view()

        line_curr.set_data(history_iter, history_current)
        ax_cost.relim()
        ax_cost.autoscale_view()

        coords = last_data["pos"].reshape((NB_CAPS, 2))
        scat_caps.set_offsets(coords)
        
        for i, txt in enumerate(annot_texts):
            txt.set_position((coords[i, 0]+1, coords[i, 1]+1))

        ax_map.set_title(f"Placement (Iter {last_data['iter']})")

    return line_gbest, line_curr, scat_caps

# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == "__main__":
    
    if not os.path.exists(EXE_PATH):
        print("Erreur: Exécutable manquant.")
        exit()

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1], figure=fig)

    ax_conv = fig.add_subplot(gs[:, 0])
    ax_cost = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[1, 1])

    line_gbest, = ax_conv.plot([], [], 'b-o', lw=2, label='Global Best')
    ax_conv.set_title("Convergence (Global Best)")
    ax_conv.grid(True)
    ax_conv.legend()

    line_curr, = ax_cost.plot([], [], 'r--s', ms=4, label='Iter Best')
    ax_cost.set_title("Coût Itération Courante")
    ax_cost.grid(True, ls='--')

    board_w, board_h = 150, 80
    ax_map.set_xlim(-board_w/2, board_w/2)
    ax_map.set_ylim(-board_h/2, board_h/2)
    
    rect_board = plt.Rectangle((-board_w/2, -board_h/2), board_w, board_h, 
                               fill=None, ec='k', lw=2, label='PCB')
    ax_map.add_patch(rect_board)
    
    rect_zone = plt.Rectangle((X_MIN, Y_MIN), AREA_SIZE, AREA_SIZE, 
                              fill=None, edgecolor='red', linestyle='--', linewidth=1.5, label='Zone Auto')
    ax_map.add_patch(rect_zone)

    ax_map.plot(20, 10, 'rx', ms=10, mew=2, label='IC (0,0 rel)')      # IC en (20, 10)
    ax_map.plot(-50, -25, 'gx', ms=10, mew=2, label='VRM')             # VRM en (-50, -25)
    
    ax_map.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax_map.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    scat_caps = ax_map.scatter([], [], c='blue', s=60, alpha=0.8, edgecolors='k')
    annot_texts = [ax_map.text(0, 0, f'C{i+1}', fontsize=8, color='blue') for i in range(NB_CAPS)]
    ax_map.set_aspect('equal')
    ax_map.legend(loc='upper left', fontsize=8)

    pso_thread = threading.Thread(target=run_pso_thread)
    pso_thread.daemon = True # Le thread mourra si on ferme la fenêtre
    pso_thread.start()

    ani = FuncAnimation(fig, update_gui, interval=500, blit=False, cache_frame_data=False)

    print(" Interface lancée. Le calcul tourne en arrière-plan...")
    plt.show()