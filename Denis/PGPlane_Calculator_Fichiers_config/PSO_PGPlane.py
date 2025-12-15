"""
Created on 08/12/2025 14:37
@author: dlespiaucq
"""

import numpy as np
import subprocess
import time
from sko.PSO import PSO
from sko.tools import set_run_mode
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

exe_path = 'PGPlane_Calculator.exe'
config_file = 'CasTest_8decap.pgconf'
data_file = 'port_file.txt'
out_File = 'out.txt'

Nb_calcul = 10

NbVarOpt = 8 #le nb de variables d'optimisation du problème
NbPopSize = 30 #Taille population = Nb particules. (valeur typ. = 10*nb variables à optimiser)
NbMaxIter = 25 #maximum number d'itérations pendant lesquelles les particules (évoluent dans l'espace des solutions.)

#bornes pour la recherche (espace des solutions)
taille_carre = 23
center_IC = [20,10]
lowerBounds= [center_IC[0]-taille_carre/2, center_IC[1]-taille_carre/2]
upperBounds= [center_IC[0]+taille_carre/2, center_IC[1]+taille_carre/2]

#Les hyperparamètres :
myW = 0.8 #inertia weight (by default 0.8)
myC1 = 0.5  #cognitive parameter (by default 0.5)
myC2 = 0.5  #social parameter (by default 0.5)

##mes propres structures de stockage des résultats :
Evo_Best_Fonction_Obj_Iter = np.zeros(NbMaxIter+1) #la meilleure éval de la fonction de cout
    #pour chaque itération de l'algo de DE.
Evo_BestX_Iter = np.zeros((NbMaxIter+1,NbVarOpt)) #on stocke à chaque itération de l'algo de DE
    #les positions du meilleur individu. 

Num_iter = 0 #pour afficher le n° de l'itération en cours
def Eval_fonction_cout(p):
    global Num_iter
    x, y = p[:, 0], p[:, 1]
    myshape = p.shape
    
    Num_iter += 1
    
    #lancement du simulation engine externe
    launch_simu_engine(x,y,1)  

    #récupération des résultats des Nb_calcul simulation
    Fonction_cout = np.loadtxt(out_File)
    
    #éval de la fonction objective : directement le résultat de la fonction évaluée
    #par l'application externe
    
    #récup de la meilleure valeur de la fonction objectif. 
    #Attention : on ne vérifie pas si la solution associée respecte bien
    #les éventuelles contraintes qu'on aurait activé.
    Evo_Best_Fonction_Obj_Iter[Num_iter-1] = min(Fonction_cout)
    print(" - Fonction cout min. = "+str(Evo_Best_Fonction_Obj_Iter[Num_iter-1])+"\n")

    #récup de la position des condensateurs conduisant à cette meilleure valeur   
    if (myshape[0] == 1):
        #print("Ind best individu = 0\n")
        Evo_BestX_Iter[Num_iter-1,:] = p[0,:]
    else:
       ind_best_individu = np.argmin(Fonction_cout)
       #print("Ind best individu = "+str(ind_best_individu)+"\n")
       Evo_BestX_Iter[Num_iter-1,:] = p[ind_best_individu,:]

    return Fonction_cout

def launch_simu_engine(x,y,write_num_iter):
    
    if np.size(x) == 1 :
        #on récupère la longueur des vecteurs x et y pourles écrire dans le fichier data_file
        Nb_calcul = 1
        print("Iteration n°"+str(Num_iter)+" - Taille x = 1")
    
        #écriture des vecteurs x et y dans le fichier data_file
        f = open(data_file, 'w') # opens the workfile file (w = write only mode)
        for ii in range(0,Nb_calcul,1):
            f.write(str(x)+" "+str(y)+"\n")        
        f.close()    
        
    else:
        #on récupère la longueur des vecteurs x et y pourles écrire dans le fichier data_file
        taille_x = np.shape(x)
        print("Iteration n°"+str(Num_iter)+" - Taille x = "+str(taille_x[0]))
        Nb_calcul = taille_x[0]
    
        #écriture des vecteurs x et y dans le fichier data_file
        f = open(data_file, 'w') # opens the workfile file (w = write only mode)
        for ii in range(0,Nb_calcul,1):
            f.write(str(x[ii])+" "+str(y[ii])+"\n")        
        f.close()
        
    #on passe les arguments au programme à appeler : 
    args = [exe_path,str(Nb_calcul),config_file,data_file,out_File]
    
    #lancement du programme externe et attente de la fin de son exécution.
    subprocess.run(args) 

#pour la mesure du temps d'exécution :
start = time.time()
 
#execute PSO search :
mode = 'vectorization'
set_run_mode(Eval_fonction_cout, mode)

myPSO = PSO(func=Eval_fonction_cout, n_dim=NbVarOpt, pop=NbPopSize, max_iter=NbMaxIter,
            lb=lowerBounds, ub=upperBounds,w=myW, c1=myC1, c2=myC2)

best_x, best_y = myPSO.run()
print('best_x:', best_x, '\n')
print('Best Obj. Function value (last iteration):',min(best_y),'\n')

stop = time.time()

#affichage du temps de calcul : 
print("Execution time subprocess.run = "+str(stop-start))

#affichage des résultats : best_x est le meilleur résultat issu de la
#dernière itération. best_y correspond à toutes les éval de la fonction objectif
#lors de la dernière itération (avec NbPopSize évaluations de la fonction objectif)
print('best_x:', best_x, '\n')
print('Best Obj. Function value (last iteration):',min(best_y),'\n')