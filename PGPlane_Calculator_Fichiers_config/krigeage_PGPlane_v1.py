import numpy as np
import subprocess
import os
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from skopt import gp_minimize
from skopt.space import Real

# =============================================================================
# 0. UTILITAIRES & CONFIG (Basé sur [cite: 35, 47, 54])
# =============================================================================

class Colors:
    HEADER, BLUE, GREEN, RED, RESET, BOLD = '\033[95m', '\033[94m', '\033[92m', '\033[91m', '\033[0m', '\033[1m'

EXE_PATH = r'PGPlane_Calculator.exe' 
CONFIG_FILE = r'CasTest_8decap.pgconf'
DATA_FILE = 'port_file.txt'
OUT_FILE = 'out.txt'

PORT_NAMES_STR = "PC1;PC2;PC3;PC4;PC5;PC6;PC7;PC8"
CAP_SIZES = [[3.2, 1.6], [2.0, 1.2], [2.0, 1.2], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [0.6, 0.3], [0.6, 0.3]] # [cite: 54]
CAP_RADII = np.array([np.sqrt(w**2 + h**2) / 2.0 for w, h in CAP_SIZES])

NB_CAPS = 8
X_CENTER, Y_CENTER = 20.0, 10.0 # [cite: 47]
AREA_SIZE = 23.0 # 
HALF_SIZE = AREA_SIZE / 2.0
X_MIN, X_MAX = X_CENTER - HALF_SIZE, X_CENTER + HALF_SIZE
Y_MIN, Y_MAX = Y_CENTER - HALF_SIZE, Y_CENTER + HALF_SIZE

# Définition de l'espace de recherche pour skopt 
SPACE = [Real(X_MIN, X_MAX, name=f'x{i//2}') if i%2==0 else Real(Y_MIN, Y_MAX, name=f'y{i//2}') for i in range(NB_CAPS * 2)]

FREQ_MIN, FREQ_MAX = 1e6, 1e9 # [cite: 19]
PTS_PER_DEC = 30 # [cite: 100]
NB_FREQ_PTS = int((np.log10(FREQ_MAX) - np.log10(FREQ_MIN)) * PTS_PER_DEC) + 1
FREQ_VECTOR = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NB_FREQ_PTS)

# Calcul de Z Target [cite: 18, 26, 27]
Z_TARGET_VECTOR = np.where(FREQ_VECTOR <= 100e6, 0.1, 0.1 * (FREQ_VECTOR / 100e6))

# =============================================================================
# 1. GESTION DES DONNÉES & MOTEUR (Basé sur [cite: 63, 137])
# =============================================================================

gui_queue = queue.Queue()
history_iter, history_current, history_gbest = [], [], []
global_iter_count = 0
global_best_so_far = float('inf')

def write_input_file_single(coords_vector):
    """ Écrit les coordonnées d'un seul individu pour l'exécutable [cite: 137] """
    with open(DATA_FILE, 'w') as f:
        f.write(PORT_NAMES_STR + "\n")
        coords_meters = np.array(coords_vector) / 1000.0
        line_str = ";".join([f"{val:.6e}" for val in coords_meters])
        f.write(line_str + "\n")

def calculate_cost(z_profile, coords_vector):
    """ Calcul du coût : violation de ZT (en log) + collisions [cite: 21, 95] """
    # Marge de 10% sur l'impédance calculée 
    z_with_margin = z_profile * 1.1
    log_diff = np.log10(z_with_margin) - np.log10(Z_TARGET_VECTOR)
    cost_z = np.sum(np.maximum(0, log_diff)**2 * np.where(FREQ_VECTOR > 100e6, 10.0, 1.0))

    # Pénalité de collision [cite: 78, 95]
    penalty_collision = 0
    coords = np.array(coords_vector).reshape(NB_CAPS, 2)
    for i in range(NB_CAPS):
        for j in range(i + 1, NB_CAPS):
            dist = np.linalg.norm(coords[i] - coords[j])
            min_dist = CAP_RADII[i] + CAP_RADII[j]
            if dist < min_dist:
                penalty_collision += ((min_dist - dist)**2) * 10
    return cost_z + penalty_collision

def obj_function(x):
    """ Fonction objectif appelée par l'Optimisation Bayésienne  """
    global global_iter_count, global_best_so_far
    
    write_input_file_single(x)
    # Appel de l'outil console pour 1 individu [cite: 137, 139]
    args = [EXE_PATH, "1", CONFIG_FILE, DATA_FILE, OUT_FILE]
    
    try:
        subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        z_profile = np.loadtxt(OUT_FILE)
    except:
        return 1e9

    cost = calculate_cost(z_profile, x)
    global_iter_count += 1
    if cost < global_best_so_far: global_best_so_far = cost

    gui_queue.put({"iter": global_iter_count, "curr": cost, "gbest": global_best_so_far, 
                   "pos": np.array(x), "z_profile": z_profile})
    
    color = Colors.GREEN if cost <= global_best_so_far else Colors.RED
    print(f" > Itération : {global_iter_count}\t| Cost : {color}{cost:.4f}{Colors.RESET}")
    return cost

def run_bayes_thread():
    """ Thread dédié au Krigeage (Gaussian Process)  """
    print(" >>> Démarrage de l'Optimisation Bayésienne...")
    res = gp_minimize(obj_function, SPACE, n_calls=50, n_initial_points=10, random_state=42)
    print(" >>> Fin du calcul Bayésien.")

# =============================================================================
# 2. INTERFACE GRAPHIQUE (MAIN THREAD)
# =============================================================================

def update_gui(frame):
    last_data = None
    while not gui_queue.empty():
        last_data = gui_queue.get_nowait()
        history_iter.append(last_data["iter"])
        history_current.append(last_data["curr"])
        history_gbest.append(last_data["gbest"])
    
    if last_data:
        line_gbest.set_data(history_iter, history_gbest)
        line_curr.set_data(history_iter, history_current)
        for ax in [ax_conv, ax_cost]: ax.relim(); ax.autoscale_view()
        
        coords = last_data["pos"].reshape((NB_CAPS, 2))
        scat_caps.set_offsets(coords)
        for i, txt in enumerate(annot_texts): txt.set_position((coords[i, 0]+0.5, coords[i, 1]+0.5))
        
        line_z_curr.set_data(FREQ_VECTOR, last_data["z_profile"])
        ax_z.relim(); ax_z.autoscale_view()
        ax_map.set_title(f"Placement Bayésien (Iter {last_data['iter']})")
    return line_gbest, line_curr, scat_caps, line_z_curr

if __name__ == "__main__":
    if not os.path.exists(EXE_PATH):
        print("Erreur: PGPlane_Calculator.exe introuvable."); exit()

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], figure=fig)
    ax_conv, ax_cost, ax_z, ax_map = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])

    line_z_target, = ax_z.loglog(FREQ_VECTOR, Z_TARGET_VECTOR, 'r--', label='Target $Z_T$')
    line_z_curr, = ax_z.loglog(FREQ_VECTOR, np.ones(NB_FREQ_PTS)*10, 'b-', label='Best $Z_{sim}$')
    ax_z.set_ylabel("Impedance ($\Omega$)"); ax_z.grid(True, which="both"); ax_z.legend()

    line_gbest, = ax_conv.plot([], [], 'b-o', label='Global Best (Krigeage)')
    line_curr, = ax_cost.plot([], [], 'r--s', ms=4, label='Iter Cost')
    for ax in [ax_conv, ax_cost]: ax.grid(True); ax.legend()

    ax_map.set_xlim(-75, 75); ax_map.set_ylim(-40, 40)
    ax_map.add_patch(plt.Rectangle((-75, -40), 150, 80, fill=None, ec='k', lw=2)) # PCB [cite: 35]
    ax_map.add_patch(plt.Rectangle((X_MIN, Y_MIN), AREA_SIZE, AREA_SIZE, fill=None, ec='r', ls='--')) # Zone [cite: 40]
    ax_map.plot(20, 10, 'rx', ms=10, label='IC'); ax_map.plot(-50, -25, 'gx', ms=10, label='VRM') # [cite: 47, 49]
    
    scat_caps = ax_map.scatter(np.zeros(NB_CAPS), np.zeros(NB_CAPS), c='blue', s=CAP_RADII*50, alpha=0.6, edgecolors='k')
    annot_texts = [ax_map.text(0, 0, f'C{i+1}', fontsize=8) for i in range(NB_CAPS)]
    ax_map.set_aspect('equal'); ax_map.legend()

    threading.Thread(target=run_bayes_thread, daemon=True).start()
    ani = FuncAnimation(fig, update_gui, interval=500, cache_frame_data=False)
    plt.show()