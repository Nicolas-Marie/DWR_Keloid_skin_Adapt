"""

Calcul de J(u) sur un maillage uniforme très fin
pour tendre vers la solution exacte,

pour ensuite calculer l'erreur relative pour les quantités d'intérêt
et tracer les courbes d'erreur, ainsi que comparer avec l'estimateur global.


Résultats issus du fichier "dwr_poisson_primal.py", voir ce fichier pour le contexte du problème

"""

"""
Auteur : Nicolas Marie
Date : 24/07/2020
Version : 1.0
Historique : 24/07/20 - Création du fichier
"""

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# maillage avec 2*(1000*1000) triangles
mesh = UnitSquareMesh(1000, 1000)

# définition de l'espace de fonction
# éléments finis P1 de Lagrange (ordre 1)
V = FunctionSpace(mesh, "P", 1)

bc = DirichletBC(V, 0.0, "near(x[0], 0.0)")

#Définition des fonctions d'essai u et test v
u = TrialFunction(V)
v = TestFunction(V)

# Terme source, second membre de l'EDP
f = Expression("-2*(x[0]*(x[0]-1) + x[1]*(x[1]-1))",degree=1)

#Définition des formes bilinéaire et linéaire de la formulation faible
a = inner(grad(u),grad(v))*dx
L = f*v*dx # condition de neumann homogène, donc pas de terme sur la frontière

#Re-définition de u pour la résolution du problème
u = Function(V)

#résolution du problème sur un maillage uniforme
solve(a == L, u, bc)


#### Calcul de la quantité d'intérêt "exacte" :

print("Calcul de la quantité d'intérêt \"exacte\" :")

### 1er exemple : intégrale sur le domaine ###
J_1 = assemble(u*dx)
print("J_1(u) = ",J_1)


### 2e exemple : intégrale sur une arête du bord ###

class ligne(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]>=0.25 and x[0]<=0.5  and near(x[1],0.0,1E-14)

arete = ligne()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)

arete.mark(boundaries, 1)

ds_sub = Measure('ds', domain=mesh, subdomain_data=boundaries, subdomain_id=1)

J_2 = assemble(u*ds_sub)
print("J_2(u) = ",J_2)


### 3e exemple : intégrale de la fonction sur un carré du maillage uniquement ###

mf = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
region = AutoSubDomain(lambda x, on: x[0] >= 0.25 and x[0] <= 0.5 and x[1] >= 0.25 and x[1]<= 0.5)
region.mark(mf, 1)
dx_sub = Measure('dx', subdomain_data=mf)

J_3 = assemble(u*dx_sub(1))
print("J_3(u) = ",J_3)



### Calcul des erreurs relatives pour chaque exemple  ###


#valeur de J_1(u_h) à chaque itération
J_1_uh = np.array([0.21066,0.220805,0.223738,0.225789,0.226681,0.227241,0.22748,0.227626,0.227694,0.227724,0.227751]) ### dans le summary de AdaptiveLinearVariationalSolver, functional value est bien J(u_h) ? ###
#nombre de cellules du maillage
nb_cells_1 = np.array([32,97,157,353,610,1361,2291,4762,8652,12133,26927])
#erreur relative
erreur_1 = abs(J_1 - J_1_uh)


#idem pour J_2(u_h)
J_2_uh = np.array([0.047206,0.0488644,0.0497328,0.0505597,0.0508049,0.0510438,0.0511644,0.051226,0.0512574])
nb_cells_2 = np.array([32,62,118,187,372,713,1361,2488,4683])
erreur_2 = abs(J_2 - J_2_uh)

#et pour J_3(u_h)
J_3_uh = np.array([0.0122378,0.0127603,0.0130652,0.0131904,0.0132369,0.0132719,0.013296,0.0133055,0.0133124,0.0133153])
nb_cells_3 = np.array([32,75,164,295,572,1099,2061,3772,7139,13045])
erreur_3 = abs(J_3 - J_3_uh)


### Estimateurs globaux pour chaque exemple ###

#Pour J_1
est_global_1 = np.array([0.003168,0.001250,0.000814,0.000443,0.000224,0.000121,0.000064,0.000036,0.000019,0.000012,0.000006])

#Pour J_2
est_global_2 = np.array([0.000801,0.000616,0.000377,0.000131,0.000097,0.000043,0.000029,0.000012,0.000008])

#Pour J_3
est_global_3 = np.array([0.000289,0.000089,0.000046,0.000021,0.000013,0.000007,0.000004,0.000002,0.000001,0.000001])


### Tracé des courbes d'erreur avec comparaison avec l'estimateur global ###

#Pour J_1
plt.figure(1)
plt.plot(nb_cells_1,erreur_1,'+',linestyle='-')
plt.plot(nb_cells_1,est_global_1,'+',linestyle='-')
plt.legend(['|J_1(u) - J_1(u_h)|','Estimateur global'])

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nombre de cellules')

plt.savefig("courbe_d_erreur_J_1.png")

#Pour J_2
plt.figure(2)
plt.plot(nb_cells_2,erreur_2,'+',linestyle='-')
plt.plot(nb_cells_2,est_global_2,'+',linestyle='-')
plt.legend(['|J_2(u) - J_2(u_h)|','Estimateur global'])

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nombre de cellules')

plt.savefig("courbe_d_erreur_J_2.png")

#Pour J_3
plt.figure(3)
plt.plot(nb_cells_3,erreur_3,'+',linestyle='-')
plt.plot(nb_cells_3,est_global_3,'+',linestyle='-')
plt.legend(['|J_3(u) - J_3(u_h)|','Estimateur global'])

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nombre de cellules')

plt.savefig("courbe_d_erreur_J_3.png")
