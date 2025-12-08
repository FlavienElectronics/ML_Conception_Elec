# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:05:51 2025

@author: adminaboyer

# > Prise en main de l'algorithme Differential Evolution (DE) de la librarie Scipy

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
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
import subprocess
import time
import matplotlib.pyplot as plt
from matplotlib import cm


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
MyBounds = Bounds(lowerBounds,upperBounds)
#MyBounds = [[-2,2] , [-2,2]]


NbPopSize = 20 #Taille population = NbPopSize*NbVarOpt (attention, si contraintes d'opt,
        #les populations qui ne respectent pas la contrainte sont éliminées).
        #valeur typ. = 10*nb variables à optimiser

NbMaxIter = 20 #maximum number of generations over which the entire population is evolved

TolAbsolu = 1e-4 #tolérance absolue pour la convergence.

#Les hyperparamètres :
proba_mut = 0.7 #probability of mutation, ou crossover proba ou recombination 
    #constant : compris entre 0 et 1, par défaut 0.3. L'augmenter créé plus de 
    #mutants qui vont progresser dans les futures générations, au risque d'une
    #instabilité dans la population
coef_mut_min = 0.5 #coefficient de mutation ou diff. weight F : compris entre 0 et 2, par défaut 0.5)
    #L'augmenter accroit le rayon de la recherche, mais ralentit la convergence.    
coef_mut_max = 0.5

#indique si on utilise ou non les contraintes :
UseConstraints = True


###############################################################################
#Définition des contraintes
###############################################################################


#définition de contraintes linéaires : par ex. -10 <= x+y <= 10
NbContraintesLin = 1
lowerBoundLinearConstraints = -2*np.ones(NbContraintesLin) #-1*np.inf*np.ones(NbVarOpt)
upperBoundLinearConstraints = 2*np.ones(NbContraintesLin)
matrixLinearConstraints = np.array([1,1])
linearConstraints = LinearConstraint(matrixLinearConstraints, lowerBoundLinearConstraints, upperBoundLinearConstraints)

#définition de contraintes nonlinéaires : par ex. 0 <= sqrt(x²+y²) <= 2
NbContraintesNonLin = 1
lowerBoundNonLinearConstraints = 0*np.ones(NbContraintesNonLin)
upperBoundNonLinearConstraints = 2*np.ones(NbContraintesNonLin)

#déf de la fonction de contraintes non-linéaires.
#Attention, dans le cas où on utilise la vectorisation, la fonction doit renvoyer
#une matrice de taille (M, S), où S est le nb de solutions du vecteurs et M est
#le nb de contraintes
def MyNonlinearConstraint(p):
    x, y = p
    if np.size(x) == 1 :
        res = np.sqrt(x**2+y**2)
    else:
        taille_x = np.shape(x)
        #print("Taille x = "+str(taille_x[0]))
        Nb_calcul = taille_x[0]   
        res = np.zeros((1,Nb_calcul))
        for ii in range(0,Nb_calcul,1):
            res[0,ii] = np.sqrt(x[ii]**2+y[ii]**2)
    
    return res
 
nonlinearConstraints = NonlinearConstraint(MyNonlinearConstraint, lowerBoundNonLinearConstraints, upperBoundNonLinearConstraints )


##mes propres structures de stockage des résultats :
Evo_Best_Fonction_Obj_Iter = np.zeros(NbMaxIter+1) #la meilleure éval de la fonction de cout
    #pour chaque itération de l'algo de DE.
Evo_BestX_Iter = np.zeros((NbMaxIter+1,NbVarOpt)) #on stocke à chaque itération de l'algo de DE
    #les positions du meilleur individu. 
   
Num_iter = 0 #pour afficher le n° de l'itération en cours


###############################################################################
#Lancement de la fonction de cout et de l'application externe (simulation engine)
###############################################################################
  

#fonction de cout, appelée par l'algo DE.  
def Eval_fonction_cout(p):
    global Num_iter
    x, y = p
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
        Evo_BestX_Iter[Num_iter-1,:] = p[:,0]
    else:
       ind_best_individu = np.argmin(Fonction_cout)
       #print("Ind best individu = "+str(ind_best_individu)+"\n")
       Evo_BestX_Iter[Num_iter-1,:] = p[:,ind_best_individu]

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
#execute differential evolution search :
###############################################################################

#pour la mesure du temps d'exécution :
start = time.time()
 

#l'utilisation de vectorized=True réduit le terme nfev (par ex d'environ 800 à 270), qui
#indique le nb d'évaluation de la fonction objectif.  
#PAr contre, sur cet exemple, la vectorization n'a pas une influence majeure sur le temps 
#d'exécution.
#polish=False  empêche l'utilisation de la méthode L-BFGS-B à l'issue des itérations 
#de l'algo génétique, qui améliore légèrement la minimisation de la solution.
#A noter que cette étape nécessite un grand nombre d'éval de la fonction objective !  
#Rq : nfev = (nit+1)*2*popsize + nb_iter_polish.
if UseConstraints:
    #avec contraintes d'inégalité :
    result = differential_evolution(Eval_fonction_cout,MyBounds,popsize=NbPopSize,atol=TolAbsolu,
                                    maxiter=NbMaxIter,mutation=(coef_mut_min,coef_mut_max),
                                    recombination=proba_mut,disp=True,vectorized=True,
                                    updating='deferred',polish=False,
                                    constraints=[linearConstraints,nonlinearConstraints])
else:    
    result = differential_evolution(Eval_fonction_cout,MyBounds,popsize=NbPopSize,atol=TolAbsolu,
                                maxiter=NbMaxIter,mutation=(coef_mut_min,coef_mut_max),
                                recombination=proba_mut,disp=True,vectorized=True,
                                updating='deferred',polish=False)
  

stop = time.time()

print(result)

#affichage du temps de calcul : 
print("Execution time subprocess.run = "+str(stop-start))


###############################################################################
#Affichage graphique des résultats
###############################################################################

#Affichage convergence de la fonction de cout à chaque itération de l'algo: tracé de 
#Evo_Best_Fonction_Obj_Iter
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

