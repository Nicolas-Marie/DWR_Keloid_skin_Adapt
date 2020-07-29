# DWR_Keloid_skin_Adapt/Poisson

Résolution du problème de Poisson par la méthode Dual-Weighted Residual (raffinement adaptatif) :

-Laplacien(u) = f

avec f = -2[x(x-1) + y(y-1)]

Domaine : Carré unité (0,1)x(0,1)

Conditions limites :
Dirichlet homogène sur la frontière gauche,
Neumannn homogène sur le reste de la frontière

Différentes quantités d'intérêt sont utilisées.


Le fichier dwr_poisson_primal.py résout le problème primal, dwr_poisson_dual.py résout le problème dual et dwr_poisson_sol_exacte.py
calcule une solution approchée sur un maillage uniforme très fin pour tendre vers la solution exacte du problème primal.

Ce dernier permet d'étudier l'erreur commise dans l'approximation de la quantité d'intérêt ainsi que l'efficacité de l'estimateur d'erreur DWR,
sous forme de courbes d'erreur.

Les valeurs numériques sont obtenues en exécutant dwr_poisson_primal.py
