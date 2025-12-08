# -*- coding: utf-8 -*-
"""
Script d'optimisation de placement de condensateurs (Power Integrity).
Algorithme : Evolution Différentielle (Scipy).
Correction : Gestion encodage Windows, Vectorisation et Fonction de coût Zpdn.
"""

import numpy as np
from scipy.optimize import differential_evolution
import subprocess
import time
import os
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMÈTRES ET CONFIGURATION
# =============================================================================

# --- CHEMINS DES FICHIERS (Vérifiez qu'ils sont dans le même dossier) ---
EXE_PATH = 'PGPlane_Calculator.exe'
CONFIG_FILE = 'CasTest_4decap.pgconf' # Mettre CasTest_8decap.pgconf pour le cas complet
PORT_FILE = 'positions_caps.txt'
RESULT_FILE = 'resultats_simulation.txt'

# --- DÉFINITION DES CONDENSATEURS ---
# Noms exacts des ports définis dans le fichier .pgconf (onglet "Port loading")
# Pour 4 condensateurs :
PORT_NAMES = ["C1", "C2", "C3", "C4"] 
NB_CAPS = len(PORT_NAMES)
NB_VAR_OPT = NB_CAPS * 2 # 2 variables (x, y) par condensateur

# --- PARAMÈTRES DE SIMULATION ---
F_MIN = 1e6     # 1 MHz
F_MAX = 1e9     # 1 GHz
NB_PTS_DEC = 30 # Points par décade

# --- ZONE DE PLACEMENT (en mètres) ---
# D'après l'énoncé : Zone 23x23mm autour de l'IC (centré en 20mm, 10mm)
X_IC, Y_IC = 20e-3, 10e-3
ZONE_WIDTH = 23e-3
X_MIN, X_MAX = X_IC - ZONE_WIDTH/2, X_IC + ZONE_WIDTH/2
Y_MIN, Y_MAX = Y_IC - ZONE_WIDTH/2, Y_IC + ZONE_WIDTH/2

# --- PARAMÈTRES ALGORITHME ---
POP_SIZE = 10   # Taille pop = 10 * nb_vars (environ 80 à 160 individus)
MAX_ITER = 20   # Nombre de générations
TOLERANCE = 1e-3

# =============================================================================
# 2. FONCTIONS UTILITAIRES (Cible et Géométrie)
# =============================================================================

def get_frequency_vector():
    """Génère le vecteur fréquence logarithmique utilisé par le simulateur"""
    nb_decades = np.log10(F_MAX) - np.log10(F_MIN)
    nb_points = int(nb_decades * NB_PTS_DEC) + 1
    return np.logspace(np.log10(F_MIN), np.log10(F_MAX), nb_points)

def get_z_target(freqs):
    """Calcule le profil Ztarget avec la marge de 10%"""
    z_target = np.ones_like(freqs) * 0.1
    mask_hf = freqs > 100e6
    # Pente +20dB/dec après 100 MHz
    z_target[mask_hf] = 0.1 * (freqs[mask_hf] / 100e6)
    return z_target * 0.9 # Marge de sécurité

def check_overlap(positions):
    """Pénalité si les condensateurs se chevauchent"""
    cap_size = 1e-3 # 1mm de côté supposé
    n = len(positions) // 2
    penalty = 0
    # Comparaison paire à paire
    for i in range(n):
        xi, yi = positions[2*i], positions[2*i+1]
        for j in range(i + 1, n):
            xj, yj = positions[2*j], positions[2*j+1]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < cap_size:
                penalty += 1e4 * (cap_size - dist) # Forte pénalité
    return penalty

# =============================================================================
# 3. INTERFAÇAGE SIMULATEUR (ROBUSTE)
# =============================================================================

def run_simulation(population):
    """
    Prépare les fichiers et lance l'exe.
    Gère la vectorisation (plusieurs individus à la fois).
    """
    # Si un seul individu (optimisation finale), on reshape
    if len(population.shape) == 1:
        population = population.reshape(-1, 1)
    
    nb_vars, nb_individus = population.shape
    
    # 1. Création du fichier de ports
    try:
        with open(PORT_FILE, 'w') as f:
            f.write(";".join(PORT_NAMES) + "\n")
            # Écriture des lignes : x1;y1;x2;y2...
            for i in range(nb_individus):
                coords = population[:, i]
                # Format scientifique pour précision
                line = ";".join([f"{val:.6e}" for val in coords])
                f.write(line + "\n")
    except Exception as e:
        print(f"Erreur écriture fichier: {e}")
        return None

    # 2. Appel de l'exécutable
    args = [EXE_PATH, str(nb_individus), CONFIG_FILE, PORT_FILE, RESULT_FILE]
    
    if not os.path.exists(EXE_PATH):
        print(f"ERREUR: {EXE_PATH} introuvable.")
        return None

    try:
        # Lancement sans 'text=True' pour éviter UnicodeDecodeError
        result = subprocess.run(args, capture_output=True)
        
        if result.returncode != 0:
            # Décodage manuel 'cp1252' (Windows FR) en ignorant les erreurs
            err = result.stderr.decode('cp1252', errors='ignore')
            print(f"Erreur simulation: {err}")
            return None
            
    except Exception as e:
        print(f"Crash subprocess: {e}")
        return None

    # 3. Lecture résultat
    try:
        if not os.path.exists(RESULT_FILE): return None
        data = np.loadtxt(RESULT_FILE)
        # Si un seul individu, s'assurer que c'est une matrice 2D (Freq, 1)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return data
    except:
        return None

# =============================================================================
# 4. FONCTION DE COÛT
# =============================================================================

freqs = get_frequency_vector()
z_target_profile = get_z_target(freqs)

def cost_function(p):
    """
    Fonction minimisée par l'algorithme.
    """
    # Adaptation format Scipy (NbVars, PopSize)
    if len(p.shape) == 1: p = p.reshape(-1, 1)
    
    nb_individus = p.shape[1]
    
    # Lancement simulation vectorisée
    z_sims = run_simulation(p)
    
    # Si échec, on renvoie un tableau de pénalités max
    if z_sims is None:
        return np.ones(nb_individus) * 1e9
    
    costs = np.zeros(nb_individus)
    
    for i in range(nb_individus):
        # Sécurité index
        if i >= z_sims.shape[1]: 
            costs[i] = 1e9
            continue

        z_curve = z_sims[:, i]
        
        # 1. Coût Électrique (Dépassement cible)
        # On ne compte que les points où Z_simu > Z_target
        depassement = np.maximum(0, z_curve - z_target_profile)
        # On somme les erreurs (pondération 100 pour donner du poids)
        cout_elec = np.sum(depassement) * 100
        
        # 2. Coût Géométrique (Chevauchement)
        coords = p[:, i]
        cout_geo = check_overlap(coords)
        
        costs[i] = cout_elec + cout_geo

    # Affichage progression (optionnel)
    best_gen = np.min(costs)
    if best_gen < 1e8:
        print(f"Generation Best Cost: {best_gen:.4f}")
    
    return costs

# =============================================================================
# 5. MAIN
# =============================================================================

if __name__ == "__main__":
    
    # 1. Définition des bornes (Bounds)
    # Liste de tuples [(min, max), (min, max)...] pour chaque variable x,y
    bounds_list = []
    for _ in range(NB_CAPS):
        bounds_list.append((X_MIN, X_MAX)) # x
        bounds_list.append((Y_MIN, Y_MAX)) # y
    
    print(f"--- DÉBUT OPTIMISATION ({NB_CAPS} condensateurs) ---")
    print(f"Bornes X: [{X_MIN*1e3:.1f}, {X_MAX*1e3:.1f}] mm")
    print(f"Bornes Y: [{Y_MIN*1e3:.1f}, {Y_MAX*1e3:.1f}] mm")
    
    t0 = time.time()
    
    # 2. Lancement Evolution Différentielle
    # vectorized=True est essentiel pour la vitesse
    res = differential_evolution(
        cost_function,
        bounds=bounds_list,
        strategy='best1bin',
        maxiter=MAX_ITER,
        popsize=POP_SIZE,
        tol=TOLERANCE,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        vectorized=True, 
        polish=False 
    )
    
    print(f"\n--- TERMINÉ en {time.time()-t0:.2f} s ---")
    print(f"Meilleur score : {res.fun}")
    print("Meilleures positions (mètres) :")
    print(res.x)

    # =============================================================================
    # 6. AFFICHAGE ET EXPORT
    # =============================================================================
    
    print("\nCalcul final pour affichage...")
    final_z = run_simulation(res.x)
    
    if final_z is not None:
        # Graphique
        plt.figure(figsize=(10, 6))
        plt.loglog(freqs, z_target_profile, 'r--', label='Z Target', linewidth=2)
        plt.loglog(freqs, final_z.flatten(), 'b-', label='Z Optimisé')
        plt.grid(True, which="both", ls="-")
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Impédance (Ohms)')
        plt.title(f'Résultat Optimisation - {NB_CAPS} Condensateurs')
        plt.legend()
        plt.show()
        
        # Export pour GUI
        # On crée un fichier formaté pour l'import dans IC-EMC
        try:
            with open("import_gui.txt", "w") as f:
                f.write(";".join(PORT_NAMES) + "\n")
                coords_str = [f"{val:.6e}" for val in res.x]
                f.write(";".join(coords_str))
            print("\nFichier 'import_gui.txt' généré.")
            print("Vous pouvez l'importer dans IC-EMC (Bouton 'Import Port coord.')")
        except:
            pass
            
    else:
        print("Erreur lors de la simulation finale.")