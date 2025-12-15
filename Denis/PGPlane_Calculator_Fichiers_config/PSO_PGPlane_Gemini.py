import numpy as np
import subprocess
import time
from sko.PSO import PSO
from sko.tools import set_run_mode
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# CONFIGURATION FICHIERS ET EXTERNES
# =============================================================================

exe_path = 'PGPlane_Calculator.exe'         # Chemin vers l'exécutable externe
config_file_base = 'CasTest_8decap.pgconf'  # Fichier de config fourni [cite: 104]
data_file = 'port_file.txt'                 # Fichier généré par ce script pour l'exe
out_File = 'out.txt'                        # Fichier résultat écrit par l'exe

# =============================================================================
# PARAMETRES DU PROBLEME (DECOUPLAGE)
# =============================================================================

# Liste des condensateurs (C1 à C8)
# Taille de sécurité (en mm) pour l'anti-collision.
# Hypothèse du TP : "Les condensateurs seront supposés carrés".
caps_info = [
    {'name': 'C1', 'size': 3.2}, # 1206: 3.2x1.6
    {'name': 'C2', 'size': 2.0}, # 0805: 2.0x1.2
    {'name': 'C3', 'size': 2.0},
    {'name': 'C4', 'size': 1.0}, # 0402: 1.0x0.5
    {'name': 'C5', 'size': 1.0},
    {'name': 'C6', 'size': 1.0},
    {'name': 'C7', 'size': 0.6}, # 0201: 0.6x0.3
    {'name': 'C8', 'size': 0.6},
]

NbCaps = len(caps_info)
NbVarOpt = NbCaps * 2            # 2 coordonnées (x, y) par condensateur

IC_X, IC_Y = 20.0, 10.0
Zone_W, Zone_H = 23.0, 23.0
x_min, x_max = IC_X - Zone_W/2, IC_X + Zone_W/2
y_min, y_max = IC_Y - Zone_H/2, IC_Y + Zone_H/2

lowerBounds = []
upperBounds = []
for _ in range(NbCaps):
    lowerBounds.extend([x_min, y_min])
    upperBounds.extend([x_max, y_max])

lowerBounds = np.array(lowerBounds)
upperBounds = np.array(upperBounds)

# =============================================================================
# PARAMETRES DE L'ALGORITHME PSO
# =============================================================================

NbPopSize = 40
NbMaxIter = 30
myW = 0.8 
myC1 = 0.5
myC2 = 0.5
UseConstraints = True

# =============================================================================
# DEFINITION DES CONTRAINTES (NON-CHEVAUCHEMENT)
# =============================================================================

constraint_ueq = []

def make_constraint(idx_i, idx_j, r_sum_sq):
    def constraint_func(x):
        xi, yi = x[2*idx_i], x[2*idx_i+1]
        xj, yj = x[2*idx_j], x[2*idx_j+1]
        
        dist_sq = (xi - xj)**2 + (yi - yj)**2
        # On veut dist_sq >= r_sum_sq  =>  r_sum_sq - dist_sq <= 0
        return r_sum_sq - dist_sq
    return constraint_func

if UseConstraints:
    for i in range(NbCaps):
        for j in range(i + 1, NbCaps):
            min_dist = (caps_info[i]['size'] + caps_info[j]['size']) / 2.0
            min_dist += 0.1 
            constraint_ueq.append(make_constraint(i, j, min_dist**2))

# =============================================================================
# STOCKAGE RESULTATS
# =============================================================================

Evo_Best_Fonction_Obj_Iter = np.zeros(NbMaxIter+1)
Evo_BestX_Iter = np.zeros((NbMaxIter+1, NbVarOpt)) 
Num_iter = 0

# =============================================================================
# CONFIGURATION SPECIFIQUE AU FICHIER DE PORTS
# =============================================================================

# Noms des ports dans l'ordre exact du fichier .pgconf
# Port 1 et 2 sont fixes (IC, VRM), les suivants sont optimisés (PC1-PC8)
port_names = ["IC", "VRM", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"]

# Coordonnées fixes extraites du XML (en mètres)
# Port 1 : IC <PortPosX> 0.02, <PortPosY> 0.01
pos_IC_fixed = [0.02, 0.01] 
# Port 2 : VRM <PortPosX> -0.05, <PortPosY> -0.025
pos_VRM_fixed = [-0.05, -0.025]

# =============================================================================
# FONCTIONS COEUR (MOTEUR DE SIMULATION)
# =============================================================================

def launch_simu_engine(population_matrix):
    """
    Génère le fichier port_file.txt au format :
    Nom1;Nom2;...
    x1;y1;x2;y2;...
    """
    
    # 1. Conversion MM -> Mètres
    # Si l'algo d'optimisation (PSO) travaille en mm (ex: 20, 10), 
    # il faut convertir en mètres pour le fichier (0.02, 0.01).
    pop_meters = population_matrix / 1000.0
    
    Nb_indiv = pop_meters.shape[0]

    # 2. Création des colonnes pour les ports fixes (IC et VRM)
    # On duplique les coordonnées fixes pour chaque individu de la population
    # Shape : (Nb_indiv, 2)
    col_IC = np.tile(pos_IC_fixed, (Nb_indiv, 1))   
    col_VRM = np.tile(pos_VRM_fixed, (Nb_indiv, 1)) 

    # 3. Assemblage de la matrice complète
    # Ordre : [IC_x, IC_y, VRM_x, VRM_y, PC1_x, PC1_y, ..., PC8_x, PC8_y]
    full_data = np.hstack((col_IC, col_VRM, pop_meters))

    # 4. Ecriture du fichier
    with open(data_file, 'w') as f:
        # [cite_start]A. Ecriture de l'en-tête (Port1;Port2...) [cite: 54, 16]
        header_str = ";".join(port_names)
        f.write(header_str + "\n")
        
        # B. Ecriture des données
        # On utilise le séparateur ";" comme demandé
        for row in full_data:
            # On formate chaque nombre pour éviter la notation scientifique si possible
            # et on joint par des points-virgules
            row_str = ";".join([f"{val:.6f}" for val in row])
            f.write(row_str + "\n")
    
    # 5. Appel du programme externe
    args = [exe_path,str(NbMaxIter),config_file_base,data_file,out_File]
    
    # Lancement silencieux (ou avec gestion d'erreur comme vu précédemment)
    try:
        print(args)
        subprocess.run(args)#, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # En cas de plantage, on peut renvoyer un fichier de coût vide ou gérer l'erreur
        pass

def Eval_fonction_cout(p):
    global Num_iter
    Num_iter += 1

    if p.ndim == 1:
        p_matrix = p.reshape(1, -1)
    else:
        p_matrix = p
        
    launch_simu_engine(p_matrix)

    try:
        Fonction_cout = np.loadtxt(out_File)
    except Exception as e:
        print(f"Erreur lecture fichier sortie : {e}")
        return 1e6 * np.ones(p_matrix.shape[0])

    if np.ndim(Fonction_cout) == 0:
        Fonction_cout = np.array([Fonction_cout])

    min_val = np.min(Fonction_cout)
    arg_min = np.argmin(Fonction_cout)
    
    Evo_Best_Fonction_Obj_Iter[Num_iter-1] = min_val
    if p.ndim == 1:
         Evo_BestX_Iter[Num_iter-1, :] = p
    else:
         Evo_BestX_Iter[Num_iter-1, :] = p[arg_min, :]

    print(f"Iter {Num_iter} | Best Cost: {min_val:.4f}")
    
    return Fonction_cout

# =============================================================================
# EXECUTION PSO
# =============================================================================

print("Démarrage de l'optimisation PSO...")
start = time.time()

set_run_mode(Eval_fonction_cout, 'vectorization')

# Instanciation PSO
pso = PSO(func=Eval_fonction_cout, n_dim=NbVarOpt, pop=NbPopSize, max_iter=NbMaxIter, 
          lb=lowerBounds, ub=upperBounds, w=myW, c1=myC1, c2=myC2,
          constraint_ueq=constraint_ueq if UseConstraints else [])

best_x, best_y = pso.run()

stop = time.time()
print(f"\nOptimisation terminée en {stop-start:.2f} secondes.")
print(f"Meilleur coût trouvé : {best_y[0]:.5f}")

# =============================================================================
# AFFICHAGE GRAPHIQUE DES RESULTATS (PLAN PCB)
# =============================================================================

# 1. Courbe de convergence
plt.figure(figsize=(8, 4))
plt.plot(pso.gbest_y_hist, color="r", linewidth=2, label="Fonction Coût")
plt.title("Convergence de l'algorithme PSO")
plt.xlabel("Générations")
plt.ylabel("Coût (Impédance)")
plt.grid(True)
plt.legend()
plt.show()

# 2. Visualisation du Placement sur le PCB [cite: 32, 38]
fig, ax = plt.subplots(figsize=(10, 6))

# Dessin de la carte (Fond Vert)
pcb_rect = patches.Rectangle((0, 0), 150, 80, linewidth=1, edgecolor='black', facecolor='#90EE90', label='PCB')
ax.add_patch(pcb_rect)

# Dessin de la Zone de Placement (Pointillés Rouges)
zone_rect = patches.Rectangle((x_min, y_min), Zone_W, Zone_H, linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Zone Placement')
ax.add_patch(zone_rect)

# Dessin des composants fixes (IC et VRM) 
ax.plot(IC_X, IC_Y, 'k+', markersize=15, markeredgewidth=3, label='IC (20,10)')
ax.text(IC_X, IC_Y+2, "IC", ha='center')

# VRM en (-50, -25) par rapport à IC ?
# Attention : Le PDF donne des coordonnées relatives sur le schéma (0,0) au centre ?
# Le schéma page 2 montre VRM à (-50, -25) et IC à (20, 10).
# Si on considère l'origine (0,0) absolue en bas à gauche de la carte 150x80 :
# Il faut probablement interpréter les coords du schéma comme relatives à un repère.
# Pour ce tracé, on utilise les coordonnées brutes optimisées qui sont dans le repère de la Zone.
# Supposons que le repère utilisé par l'optimiseur (x,y) correspond au repère du schéma.

# Dessin des Condensateurs Optimisés
# best_x contient [x1, y1, x2, y2, ...]
colors = ['blue', 'blue', 'blue', 'orange', 'orange', 'orange', 'purple', 'purple'] # Différencier par taille si on veut

for i in range(NbCaps):
    cx = best_x[2*i]
    cy = best_x[2*i+1]
    sz = caps_info[i]['size']
    name = caps_info[i]['name']
    
    # On dessine un carré centré sur (cx, cy)
    rect = patches.Rectangle((cx - sz/2, cy - sz/2), sz, sz, linewidth=1, edgecolor='black', facecolor='blue')
    ax.add_patch(rect)
    ax.text(cx, cy, name, color='white', fontsize=8, ha='center', va='center')

ax.set_xlim(0, 150)
ax.set_ylim(0, 80)
# Zoomer un peu sur la zone intéressante si nécessaire
# ax.set_xlim(x_min - 10, x_max + 10)
# ax.set_ylim(y_min - 10, y_max + 10)

plt.title("Placement Optimisé des Condensateurs")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.legend(loc='upper right')
plt.axis('equal') # Pour garder les proportions carrées
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print("\nCoordonnées finales (X, Y) :")
for i in range(NbCaps):
    print(f"{caps_info[i]['name']} : ({best_x[2*i]:.3f}, {best_x[2*i+1]:.3f})")