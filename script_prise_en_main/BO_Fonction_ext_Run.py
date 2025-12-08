# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 12:05:51 2025

@author: adminaboyer

On teste l'algo de bayesian optimization de Scikit-optimize avec :'
 
 > Prise en main de l'algorithme bayesian optimization de Scikit-optimize

 > On utilise un prog externe pour calculer la fonction objective.
 Cela passe par une fenêtre de commande, avec l'utilisation de subprocess.run.

 > on ne peut pas définir de contraintes formellement. Mais on tente de les ajouter
 dans la fonction objectif 

 > Pas de vectorisation possible (échantillonnage adaptatif).

 > Simulation engine = Calcul_Fonction_Obj.exe. Renvoie le résultat de la fonction
 sqrt(Sqrt(sqr(X[i]-X0) + sqr(Y[i]-Y0))), où (X0;Y0) est le minimum de la fonction 
 et X[i] et Y[i] sont les 2 paramètres d'entrée de la fonction. Dans le cas où la
 vectorisation est activée, i est compris entre 1 et NP (la taille de la population).

"""

import numpy as np
import subprocess
import time
from skopt import gp_minimize
import matplotlib.pyplot as plt

#lien vers l'executable calculant la fonction objectif, avec passage de ses arguments
#A actualiser selon l'emplacement des fichiers !!!
#Attention : pas d'espace dans les noms de fichier !!!
exe_path = 'Calcul_Fonction_Obj.exe' 
data_file = 'data_file.txt'  
Out_File = 'out.txt'


###############################################################################
#paramètres du problème à optimiser :
###############################################################################

#L'exemple utilisé : on cherche le minimum de la fonction :
#    sqrt(Sqrt(sqr(X[i]-X0) + sqr(Y[i]-Y0)))
#Par défaut, X0 = Y0 = 0. Mais on peut passer ces 2 valeurs en paramètres
#au programme exe appelé (params 4 et 5 de ).
X0 = 1
Y0 = -1

# #on trace la surface de réponse de la fonction à optimiser :
# #define ranges 
# x_range = np.arange(-2,2,0.05)
# y_range = np.arange(-2,2,0.05)

# #create mesh grid
# x, y = np.meshgrid(x_range,y_range)

# #define the objective function (with local minima)
# def f1(x,y):
#     r = np.sqrt(np.sqrt((x-X0)**2 + (y-Y0)**2))
#     return r

# z = f1(x,y)

# #plot the surface
# fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
# surf = ax.plot_surface(x,y,z,cmap=cm.jet,linewidth=0,antialiased=False)
# plt.show()

###############################################################################
#paramètres pour l'algo d'optimisation : 
###############################################################################
     
NbVarOpt = 2 #le nb de variables d'optimisation du problème
    
#bornes pour la recherche
MyBounds = [(-2.0, 2.0),(-2.0,2.0)]

#les paramètres de l'outil d'optimisation
NbMaxIter = 80 #maximum number of iterations of gaussian optimization
NbInitPts = 20 #le nb d'éval de la fonction à évaluer avant de l'approximer avec le modèle de process gaussien
myInitPtGenerator = "lhs" #random","sobol","halton","hammersly","lhs" 
myAcqFunc = "LCB" #"LCB", "EI", "PI""gp_hedge","EIps", "PIps"
MyAcqOptimizer = "lbfgs"  #"sampling" or "lbfgs"
NbPtsSampling = 20 #Number of points to sample to determine the next “best” point. Useless if acq_optimizer is set to "lbfgs".
NbRestartLBFGS = 4 #The number of restarts of the optimizer when acq_optimizer is "lbfgs".
MyKappa = 3 #Controls how much of the variance in the predicted values should be taken into account. If set to be very high, then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "LCB".
MyXi = 0.01 #Controls how much improvement one wants over the previous best values. Used when the acquisition is either "EI" or "PI".
NoiseLevel = 1e-10 #Set this to a value close to zero (1e-10) if the function is noise-free. Setting to zero might cause stability issues.


###############################################################################
#Définition des contraintes
###############################################################################

#les points doivent rester à l'intérieur d'un cercle de rayon Rmax
Rmax = 2
penalite = 4 #la valeur renvoyée par la fonction de cout quand la contrainte n'est pas 
 #respectée.


Num_iter = 0


###############################################################################
#Lancement de la fonction de cout et de l'application externe (simulation engine)
###############################################################################
  
#fonction de cout, appelée par l'algo BO.  
def Eval_fonction_cout(p):
    x, y = p[0], p[1]   
    
    #lancement du simulation engine externe
    launch_simu_engine(x,y,1)  

    #récupération des résultats des Nb_calcul simulation
    r = np.loadtxt(Out_File)
    Fonction_cout = r + 0 #si je renvoie directement r, il m'indique que la fonction doit
     #renvoyer un scalaire
     
     #ajout de la containte : on pénalise la fonction objectif si la contrainte
    #n'est pas respectée. Pour cela, on fixe arbitrairement une valeur max
    #(plus grande que le minimum recherché) quand la contrainte n'est pas respectée.
    distance = np.sqrt((x-X0)**2+(y-Y0)**2)
    if distance > Rmax:  #violation de la contrainte
        Fonction_cout = penalite #attention à ne pas mettre une valeur trop grande, qui dégrade le
            #modèle GPR extrait.
     
    print(" - Fonction cout = "+str(Fonction_cout)+"\n")

    return Fonction_cout


#cette fonction lance l'exécution de l'application externe Calcul_Fonction_Obj.exe.
#Elle prépare le fichier de données d'entrée (data_file) et les arguments
#pour le lancement de l'application.
#write_num_iter autorise l'affichage du n° de l'itération si on passe la valeur 1.  
def launch_simu_engine(x,y,write_num_iter):
    
    global Num_iter
    Num_iter += 1 
    
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
    args = [exe_path,str(Nb_calcul),data_file,Out_File,str(X0),str(Y0)]
    
    #lancement du programme externe et attente de la fin de son exécution.
    subprocess.run(args) 


###############################################################################
#execute BO search :
###############################################################################

start = time.time()
 
#execute gaussian process :
res = gp_minimize(Eval_fonction_cout,    # the function to minimize
                  #si les bornes sont écrites sous la forme d'un entier, alors l'algo
                  #ne tire que des entiers --> ne pas écrire 2 mais 2.0
                  MyBounds,      # the bounds on each dimension of x
                  acq_func=myAcqFunc,      # the acquisition function
                  kappa = MyKappa,
                  xi = MyXi,
                  n_calls=NbMaxIter,         # the number of evaluations of f
                  #n_random_starts=8,  # the number of random initialization points --> Deprecated since version 0.8: use n_initial_points instead.
                  n_initial_points = NbInitPts, #Number of evaluations of func with initialization points before approximating it with base_estimator
                  initial_point_generator=myInitPtGenerator,
                  acq_optimizer=MyAcqOptimizer,
                  n_points = NbPtsSampling,
                  n_restarts_optimizer = NbRestartLBFGS,
                  noise=NoiseLevel,       # the noise level (optional)
                  verbose = False,
                  random_state=1234)   # the random seed : Set random state to something other than None for reproducible results



print('best_x:', res.x[0]," ; ",res.x[1], '\n')
print('Best Obj. Function value :',res.fun,'\n')


stop = time.time()

#affichage du temps de calcul : 
print("Execution time subprocess.run = "+str(stop-start))

#affichage convergence
from skopt.plots import plot_convergence
plot_convergence(res)
plt.show()

# Plot f(x) + contours --> valable en 1D
# from skopt.plots import plot_gaussian_process
#  _ = plot_gaussian_process(res, objective=MyFunc,
#                           noise_level=0)

# plt.show()

# Plot f(x) + contours --> valable en 2D
from skopt.plots import plot_objective

_ = plot_objective(res)
plt.show()