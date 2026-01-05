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

EXE_PATH = r'PGPlane_Calculator.exe' 
CONFIG_FILE = r'CasTest_8decap.pgconf'
DATA_FILE = 'port_file.txt'
OUT_FILE = 'out.txt'

PORT_NAMES_STR = "PC1;PC2;PC3;PC4;PC5;PC6;PC7;PC8"
CAP_1206 = [3.2, 1.6]  # Tailles boitiers 1206
CAP_0805 = [2.0, 1.2]  # Tailles boitiers 0805
CAP_0402 = [1.0, 0.5]  # Tailles boitiers 0402
CAP_0201 = [0.6, 0.3]  # Tailles boitiers 0201

# Assignation selon votre budget : C1(1206), C2-3(0805), C4-6(0402), C7-8(0201)
CAP_SIZES = [
    CAP_1206,                     # C1
    CAP_0805, CAP_0805,           # C2, C3
    CAP_0402, CAP_0402, CAP_0402, # C4, C5, C6
    CAP_0201, CAP_0201            # C7, C8
]

# Calcul du rayon de sécurité pour chaque capa (moitié de la diagonale)
CAP_RADII = np.array([np.sqrt(w**2 + h**2) / 2.0 for w, h in CAP_SIZES])

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

# Fonction de coup linéaire (différence entre target et simulé)
def calculate_cost_from_z(z_matrix, population_matrix):
    z_target_col = Z_TARGET_VECTOR.reshape(-1, 1)
    log_diff = np.log10(z_matrix) - np.log10(z_target_col)
    penalty_z = np.maximum(0, log_diff)**2
    
    weights = np.where(FREQ_VECTOR > 100e6, 10.0, 1.0).reshape(-1, 1)
    cost_impedance = np.sum(penalty_z * weights, axis=0)

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
                    overlap = min_dist - dist
                    penalty_n += (overlap**2) * 5000
        
        collision_penalties[n] = penalty_n

    return cost_impedance + collision_penalties

# Foncrion de coût avec pondération en fréquence
# def calculate_cost_from_z(z_matrix):
#     z_target_col = Z_TARGET_VECTOR.reshape(-1, 1)
#     diff = z_matrix - z_target_col
#     penalty_matrix = np.maximum(0, diff)
#     weights = np.ones_like(FREQ_VECTOR)
#     mask_hf = FREQ_VECTOR > 100e6
#     weights[mask_hf] = 1.0 + (FREQ_VECTOR[mask_hf] - 100e6) / (1e9 - 100e6) * 9.0   # Pondération entre 100 M et 1 GHz
#     weighted_penalty = penalty_matrix * weights.reshape(-1, 1)
    
#     return np.sum(weighted_penalty, axis=0)

# Fonction de coput Intégrale de l'Erreur Pondérée (RMSE Fréquentiel)
# def calculate_cost_from_z(z_matrix):
#     log_diff = np.log10(z_matrix) - np.log10(Z_TARGET_VECTOR.reshape(-1, 1)) # Passage en log pour normaliser les ordres de grandeur
#     penalty = np.maximum(0, log_diff)**2 # Carré pour punir sévèrement les gros écarts
#     weights = np.where(FREQ_VECTOR > 100e6, 10.0, 1.0).reshape(-1, 1) # On définit des poids : 1 en BF, 10 en HF (au delà de 100MHz)
#     cost = np.trapz(penalty * weights, x=np.log10(FREQ_VECTOR), axis=0) # Intégration par la méthode des trapèzes sur l'axe log de la fréquence
#     return cost

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
    if z_matrix is None: return np.ones(p.shape[0]) * 1e9
    
    costs = calculate_cost_from_z(z_matrix, p)
    current_min = np.min(costs)
    best_idx = np.argmin(costs)
    current_best_pos = p[best_idx, :]
    current_best_z = z_matrix[:, best_idx] 

    global_iter_count += 1
    if current_min < global_best_so_far:
        global_best_so_far = current_min

    data_packet = {
        "iter": global_iter_count,
        "curr": current_min,
        "gbest": global_best_so_far,
        "pos": current_best_pos,
        "z_profile": current_best_z  # <--- Nouveau
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

        if "z_profile" in last_data:
            line_z_curr.set_data(FREQ_VECTOR, last_data["z_profile"])
            ax_z.relim()
            ax_z.autoscale_view()

    return line_gbest, line_curr, scat_caps, line_z_curr

# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == "__main__":
    
    if not os.path.exists(EXE_PATH):
        print("Erreur: Exécutable manquant.")
        exit()

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1], figure=fig)

    ax_conv = fig.add_subplot(gs[0, 0])
    ax_cost = fig.add_subplot(gs[0, 1])
    ax_z    = fig.add_subplot(gs[1, 0])
    ax_map  = fig.add_subplot(gs[1, 1])

    line_z_target, = ax_z.loglog(FREQ_VECTOR, Z_TARGET_VECTOR, 'r--', label='Target Z')
    line_z_curr, = ax_z.loglog(FREQ_VECTOR, np.zeros(NB_FREQ_PTS), 'b-', label='Current Best Z')
    ax_z.set_title("Impédance vs Fréquence")
    ax_z.set_xlabel("Fréquence (Hz)")
    ax_z.set_ylabel("Z (Ohms)")
    ax_z.grid(True, which="both", ls="-", alpha=0.5)
    ax_z.legend()

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
    
    scat_caps = ax_map.scatter(np.zeros(NB_CAPS), np.zeros(NB_CAPS), c='blue', s=CAP_RADII*100, alpha=0.7, edgecolors='k', zorder=5)
    annot_texts = [ax_map.text(0, 0, f'C{i+1}', fontsize=8, color='blue') for i in range(NB_CAPS)]
    ax_map.set_aspect('equal')
    ax_map.legend(loc='upper left', fontsize=8)

    pso_thread = threading.Thread(target=run_pso_thread)
    pso_thread.daemon = True # Le thread mourra si on ferme la fenêtre
    pso_thread.start()

    ani = FuncAnimation(fig, update_gui, interval=500, blit=False, cache_frame_data=False)

    print(" Interface lancée. Le calcul tourne en arrière-plan...")
    plt.show()