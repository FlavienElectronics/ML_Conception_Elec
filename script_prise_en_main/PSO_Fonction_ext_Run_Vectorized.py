# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:05:51 2025

@author: adminaboyer

> Prise en main de l'algorithme PSO de la librarie Scikit-opt

> On utilise un prog externe pour calculer la fonction objective.
Cela passe par une fenêtre de commande, avec l'utilisation de subprocess.run.

> Pour résoudre le goulet d'étranglement que constitue la gestion des processus, 
on exploite la vectorisation proposée par la fonction diff_evolution.

> Simulation engine = Calcul_Fonction_Obj.exe. Renvoie le résultat de la fonction
sqrt(Sqrt(sqr(X[i]-X0) + sqr(Y[i]-Y0))), où (X0;Y0) est le minimum de la fonction 
et X[i] et Y[i] sont les 2 paramètres d'entrée de la fonction. Dans le cas où la
vectorisation est activée, i est compris entre 1 et NP (la taille de la population).

"""



import numpy as np
import subprocess
import time
from sko.PSO import PSO
from sko.tools import set_run_mode
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

plt.ion()


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
    
#bornes pour la recherche (espace des solutions)
lowerBounds= -2*np.ones(NbVarOpt) 
upperBounds= 2*np.ones(NbVarOpt)


NbPopSize = 30 #Taille population = Nb particules.
        #valeur typ. = 10*nb variables à optimiser

NbMaxIter = 25 #maximum number d'itérations pendant lesquelles les particules 
    #évoluent dans l'espace des solutions.

#Les hyperparamètres :
myW = 0.8 #inertia weight (by default 0.8)
myC1 = 0.5  #cognitive parameter (by default 0.5)
myC2 = 0.5  #social parameter (by default 0.5)

#indique si on utilise ou non les contraintes :
UseConstraints = True


###############################################################################
#Définition des contraintes
###############################################################################


#définition d'une contrainte d'inégalité :
#   sqrt(x²+y²) <= R
#Ca se traduit sous la forme : sqrt(x²+y²) - R <= 0
constraint_ueq = [
    lambda x: np.sqrt(x[0]**2 + x[1]**2)-2
]


##mes propres structures de stockage des résultats :
Evo_Best_Fonction_Obj_Iter = np.zeros(NbMaxIter+1) #la meilleure éval de la fonction de cout
    #pour chaque itération de l'algo de DE.
Evo_BestX_Iter = np.zeros((NbMaxIter+1,NbVarOpt)) #on stocke à chaque itération de l'algo de DE
    #les positions du meilleur individu. 
   
Num_iter = 0 #pour afficher le n° de l'itération en cours


###############################################################################
#Lancement de la fonction de cout et de l'application externe (simulation engine)
###############################################################################
  

#fonction de cout, appelée par l'algo PSO.  
def Eval_fonction_cout(p):
    global Num_iter
    x, y = p[:, 0], p[:, 1]
    myshape = p.shape
    
    Num_iter += 1
    
    #lancement du simulation engine externe
    launch_simu_engine(x,y,1)  

    #récupération des résultats des Nb_calcul simulation
    Fonction_cout = np.loadtxt(Out_File)
    
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


#cette fonction lance l'exécution de l'application externe Calcul_Fonction_Obj.exe.
#Elle prépare le fichier de données d'entrée (data_file) et les arguments
#pour le lancement de l'application.
#write_num_iter autorise l'affichage du n° de l'itération si on passe la valeur 1.  
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
    args = [exe_path,str(Nb_calcul),data_file,Out_File,str(X0),str(Y0)]
    
    #lancement du programme externe et attente de la fin de son exécution.
    subprocess.run(args) 

###############################################################################
#execute PSO search :
###############################################################################

#pour la mesure du temps d'exécution :
start = time.time()
 

#execute PSO search :
mode = 'vectorization'
set_run_mode(Eval_fonction_cout, mode)


if UseConstraints:
    myPSO = PSO(func=Eval_fonction_cout, n_dim=NbVarOpt, pop=NbPopSize, max_iter=NbMaxIter, 
            lb=lowerBounds, ub=upperBounds,w=myW, c1=myC1, c2=myC2,
            constraint_ueq=constraint_ueq)  
else:
    myPSO = PSO(func=Eval_fonction_cout, n_dim=NbVarOpt, pop=NbPopSize, max_iter=NbMaxIter, 
                 lb=lowerBounds, ub=upperBounds,w=myW, c1=myC1, c2=myC2)    

myPSO.record_mode = True   #pour activer l'enregistrement --> animation (prend du temps)

#lancement de l'algo PSO. best_x est le meilleur résultat issu de la
#dernière itération. best_y correspond à toutes les éval de la fonction objectif
#lors de la dernière itération (avec NbPopSize évaluations de la fonction objectif)
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


###############################################################################
#Affichage graphique des résultats
###############################################################################

#affichage convergence de la fonction de cout à chaque génération: tracé de generation_best_Y
Eval = np.linspace(1, NbMaxIter, NbMaxIter)
plt.plot(Eval,myPSO.gbest_y_hist, color="red", linewidth=2.5, linestyle="-", label="Fonction objectif") 
plt.grid()
#plt.legend(loc='upper right')
plt.title("Evolution gbest_y_hist à chaque génération")
plt.xlabel('Nb génération', color='black')
plt.ylabel('Fonction de cout', color='black')
plt.show()

#affichage convergence de la fonction de cout à chaque itération de l'algo: tracé de 
#ce que j'ai stocké : Evo_Best_Fonction_Obj_Iter
Eval = np.linspace(1, Num_iter, Num_iter)
plt.plot(Eval,Evo_Best_Fonction_Obj_Iter, color="red", linewidth=2.5, linestyle="-", label="Fonction objectif") 
plt.grid()
#plt.legend(loc='upper right')
plt.title("Evolution min. fonction objectif à chaque itération")
plt.xlabel('Nb Itération', color='black')
plt.ylabel('Fonction de cout', color='black')
plt.show()


#Tracé de l'évolution de la position de l'optimum 
indBest = np.argmin(Evo_Best_Fonction_Obj_Iter)
X_best = np.zeros((NbMaxIter+1))
Y_best = np.zeros((NbMaxIter+1))
for ee in range (0,NbMaxIter+1,1):
    X_best[ee] = Evo_BestX_Iter[ee,0]    
    Y_best[ee] = Evo_BestX_Iter[ee,1]

plt.scatter(X_best,Y_best, color="blue", linewidth=2.5, linestyle="-", label="Optimum")  
plt.scatter(X_best[indBest],Y_best[indBest], color="orange", linewidth=7.5, linestyle="-", label="Optimum")  

plt.grid()
#plt.legend(loc='upper right')
plt.title("Evolution position optimum à chaque génération")
plt.xlabel('X', color='black')
plt.ylabel('Y', color='black')
plt.show()

#animation - attention : instructif mais prend du temps à construire !!!
# if myPSO.record_mode:
#     record_value = myPSO.record_value
#     X_list, V_list = record_value['X'], record_value['V']
    
#     fig, ax = plt.subplots(1, 1)
#     ax.set_title('title', loc='center')
#     line = ax.plot([], [], 'b.')
    
#     def MyFunc(x,y):
#         return np.sqrt(np.sqrt((x-X0)**2+(y-Y0)**2))
    
#     X_grid, Y_grid = np.meshgrid(np.linspace(-2.0, 2.0, 40), np.linspace(-2.0, 2.0, 40))
#     Z_grid = MyFunc(X_grid, Y_grid)
#     ax.contour(X_grid, Y_grid, Z_grid, 30)
        
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)

    
#     plt.ion()
#     p = plt.show()
    
#     def update_scatter(frame):
#         i, j = frame // 10, frame % 10
#         ax.set_title('iter = ' + str(i))
#         X_tmp = X_list[i] + V_list[i] * j / 10.0
#         plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
#         return line
    
#     ani = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=NbMaxIter * 10)
#     plt.show()
    
#     ani.save('pso_result.gif', writer='pillow')