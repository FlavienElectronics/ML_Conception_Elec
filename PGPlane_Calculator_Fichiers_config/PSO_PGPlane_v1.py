import numpy as np
import subprocess
import time
import os
from sko.PSO import PSO
from sko.tools import set_run_mode
import matplotlib.pyplot as plt

# =============================================================================
# 0. UTILITAIRES
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# =============================================================================
# 1. CONFIGURATION DES CHEMINS ET FICHIERS
# =============================================================================

EXE_PATH = r'PGPlane_Calculator.exe' 
CONFIG_FILE = r'CasTest_8decap.pgconf'
DATA_FILE = 'port_file.txt'
OUT_FILE = 'out.txt'

PORT_NAMES_STR = "PC1;PC2;PC3;PC4;PC5;PC6;PC7;PC8"

# =============================================================================
# 2. PARAMÈTRES DU PROBLÈME ET DE LA CIBLE
# =============================================================================

NB_CAPS = 8                                 # Nombre de condensateurs à optimiser
NB_VAR_OPT = NB_CAPS * 2                    # Nombre de variables (x et y pour chaque condensateur)

X_CENTER, Y_CENTER = 20.0, 10.0             # Centre de la zone d'optimisation (autour de l'IC)
AREA_SIZE = 23.0
HALF_SIZE = AREA_SIZE / 2.0
X_MIN, X_MAX = X_CENTER - HALF_SIZE, X_CENTER + HALF_SIZE
Y_MIN, Y_MAX = Y_CENTER - HALF_SIZE, Y_CENTER + HALF_SIZE

LOWER_BOUNDS = []                                       # Création des bornes pour le PSO
UPPER_BOUNDS = []                                       # (format [x1_min, y1_min, x2_min, y2_min, ...])
for _ in range(NB_CAPS):
    LOWER_BOUNDS.extend([X_MIN, Y_MIN])
    UPPER_BOUNDS.extend([X_MAX, Y_MAX])

FREQ_MIN = 1e6                                          # Début de la bande fréquence
FREQ_MAX = 1e9                                          # Fin de la bande fréquence
PTS_PER_DEC = 30                                        # Points par décade
NB_DECADE = np.log10(FREQ_MAX) - np.log10(FREQ_MIN)     # Nombre de décades
NB_FREQ_PTS = int(NB_DECADE * PTS_PER_DEC) + 1          # Nombre total de points fréquence
FREQ_VECTOR = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NB_FREQ_PTS)

Z_TARGET_VECTOR = np.zeros(NB_FREQ_PTS)                 # Construction du vecteur Z_TARGET (Impédance cible)
F_CORNER = 100e6                                        # Fréquence de coin
Z_FLAT = 0.1                                            # Règle : 0.1 Ohm jusqu'à 100 MHz, puis +20dB/dec

for i, f in enumerate(FREQ_VECTOR):
    if f <= F_CORNER:
        Z_TARGET_VECTOR[i] = Z_FLAT
    else:
        Z_TARGET_VECTOR[i] = Z_FLAT * (f / F_CORNER)    # +20dB/dec revient à dire Z proportionnel à f

# =============================================================================
# 3. PARAMÈTRES PSO
# =============================================================================

NB_POP_SIZE = 40                # Taille de l'essaim
NB_MAX_ITER = 30                # Nombre d'itérations
W_INERTIA = 0.8                 # Inertie
C1_COGNITIVE = 0.5              # Paramètre cognitif
C2_SOCIAL = 0.5                 # Paramètre social

history_best_cost = []          # Stockage pour graphiques : coût minimum par itération
history_best_pos = []           # Stockage pour graphiques : position du meilleur par itération

global_iter_count = 0           # Compteur global

# =============================================================================
# 4. FONCTIONS UTILITAIRES
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
    nb_individuals = z_matrix.shape[1]
    costs = np.zeros(nb_individuals)
    z_target_col = Z_TARGET_VECTOR.reshape(-1, 1)           # compare Z_simulé vs Z_target pour chaque fréquence
    
    diff = z_matrix - z_target_col                          # Calcul de la différence : Z_simu - Z_target
    
    penalty_matrix = np.maximum(0, diff)                    # Pénalité que si Z_simu > Z_target (diff > 0)
    
    costs = np.sum(penalty_matrix, axis=0)                  # # Le coût est la somme des pénalités sur toutes les fréquences
    
    return costs

def launch_simu_engine(population_matrix):
    global global_iter_count
    nb_individuals = population_matrix.shape[0]
    write_input_file(population_matrix)

    args = [
        EXE_PATH,
        str(nb_individuals),
        CONFIG_FILE,
        DATA_FILE,
        OUT_FILE
    ]

    try:
        subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Erreur critique lors du lancement : {e}")
        return None

    if os.path.exists(OUT_FILE):
        try:
            z_results = np.loadtxt(OUT_FILE)
            return z_results
        except Exception as e:
            print(f"Erreur lecture résultats (fichier corrompu ?) : {e}")
            return None
    else:
        print("Erreur : Le fichier de sortie n'a pas été créé par le solveur.")
        return None

# =============================================================================
# 5. FONCTION OBJECTIF (Appelée par PSO)
# =============================================================================

def obj_function(p):
    global global_iter_count
    global_iter_count += 1

    z_matrix = launch_simu_engine(p)
    
    if z_matrix is None:
        print("Warning: Simulation failed to return data.")
        return np.ones(p.shape[0]) * 1e9
    
    costs = calculate_cost_from_z(z_matrix)
    
    min_cost = np.min(costs)
    best_idx = np.argmin(costs)
    if min_cost <= history_best_cost[-1]:
        print(f" > Itération =\t{global_iter_count}\t{Colors.BOLD}|{Colors.RESET} Min Cost =\t{Colors.GREEN}{min_cost:.4f}{Colors.RESET}")
    else:
        print(f" > Itération =\t{global_iter_count}\t{Colors.BOLD}|{Colors.RESET} Min Cost =\t{Colors.RED}{min_cost:.4f}{Colors.RESET}")

    history_best_cost.append(min_cost)
    history_best_pos.append(p[best_idx, :])
    
    return costs

# =============================================================================
# 6. EXÉCUTION DU MAIN
# =============================================================================

if __name__ == "__main__":
    
    print(f"=== Démarrage Optimisation PSO Découplage ({NB_CAPS} Capas) ===")
    print(f"Population: {NB_POP_SIZE}, Iterations: {NB_MAX_ITER}")
    print(f"Variables: {NB_VAR_OPT} variables ({NB_CAPS} condensateurs)")
    
    # Vérification présence fichiers
    if not os.path.exists(EXE_PATH):
        print(f"ERREUR: {EXE_PATH} introuvable.")
        exit()
    if not os.path.exists(CONFIG_FILE):
        print(f"ERREUR: {CONFIG_FILE} introuvable.")
        exit()

    start_time = time.time()

    set_run_mode(obj_function, 'vectorization')             # Configuration du mode vectorisation pour scikit-opt

    pso = PSO(func=obj_function,                            # Initialisation PSO
              n_dim=NB_VAR_OPT, 
              pop=NB_POP_SIZE, 
              max_iter=NB_MAX_ITER, 
              lb=LOWER_BOUNDS, 
              ub=UPPER_BOUNDS, 
              w=W_INERTIA, 
              c1=C1_COGNITIVE, 
              c2=C2_SOCIAL)

    best_x, best_y = pso.run()                              # Lancement de l'optimisation

    end_time = time.time()
    
    print("\n=== Fin Optimisation ===")
    print(f"Temps exécution : {end_time - start_time:.2f} s")
    print(f"Meilleur Coût : {best_y}")
    print("Meilleure configuration (x1, y1, x2, y2 ...) [mm]:")
    print(np.round(best_x, 2))

    # =========================================================================
    # 7. AFFICHAGE GRAPHIQUE
    # =========================================================================
    
    # Tracer la convergence
    plt.figure(figsize=(10, 5))
    plt.plot(pso.gbest_y_hist, color='b', linewidth=2, label='Meilleur coût global')
    plt.xlabel('Itérations')
    plt.ylabel('Fonction de Coût (Dépassement Z)')
    plt.title('Convergence PSO - Placement Condensateurs')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Visualisation du placement final
    plt.figure(figsize=(8, 5))
    plt.xlim(0, 150)
    plt.ylim(0, 80)
    plt.axvline(0, color='k'); plt.axvline(150, color='k')
    plt.axhline(0, color='k'); plt.axhline(80, color='k')
    
    plt.plot(20, 10, 'rx', markersize=10, label='IC (Cible)')
    plt.plot(50, 25, 'gx', markersize=10, label='VRM') 
    
    # Dessin des condensateurs optimisés
    coords = best_x.reshape((NB_CAPS, 2))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=50, label='Condensateurs')
    
    # Annotations
    for i in range(NB_CAPS):
        plt.text(coords[i, 0], coords[i, 1], f'PC{i+1}', fontsize=9)

    plt.title('Placement final optimisé')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.axis('equal')
    plt.show()