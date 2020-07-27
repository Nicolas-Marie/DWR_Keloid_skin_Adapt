"""
Résolution du problème de Poisson
par la méthode Dual-Weighted Residual (raffinement adaptatif) :

-Laplacien(u) = f

avec f = -2[x(x-1) + y(y-1)]

Domaine : Carré unité (0,1)x(0,1)

Conditions limites :
Dirichlet homogène sur la frontière gauche,
Neumannn homogène sur le reste de la frontière
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

#Choix de l'algorithme de raffinement
parameters["refinement_algorithm"]="plaza_with_parent_facets"

# construction du maillage initial
# maillage uniforme avec 4x4 carrés divisés chacun en 2 triangles
mesh = UnitSquareMesh(4, 4)

# définition de l'espace de fonction
# éléments finis P1 de Lagrange (ordre 1)
V = FunctionSpace(mesh , "P", 1)

#Définition des fonctions d'essai et test
u = TrialFunction(V)
v = TestFunction(V)

# Terme source, second membre de l'EDP
f = Expression("-2*(x[0]*(x[0]-1) + x[1]*(x[1]-1))",degree=1)

#Définition des formes bilinéaire et linéaire de la formulation faible
a = inner(grad(u),grad(v))*dx
L = f*v*dx # pas de terme sur la frontière dans l'expression car condition de neumann homogène

# définition de la condition de Dirichlet
bc = DirichletBC(V, 0.0, "near(x[0], 0.0)")

# redéfinition de la fonction u pour la résolution du problème
u = Function(V)

"""
Définitions de plusieurs exemples de quantités d'intérêt
"""

###1er exemple : intégrale sur le domaine ###
def J_1(w):
    return w*dx


### 2e exemple : intégrale sur une arête du bord ###

#Création d'une classe pour préciser l'arête qui nous intéresse
class ligne(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]>=0.25 and x[0]<=0.5  and near(x[1],0.0,1E-14)

arete = ligne()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)

#marque l'arête qui nous intéresse
arete.mark(boundaries, 1)

#définition de la mesure pour intégrer seulement sur une partie de la frontière
ds_sub = Measure('ds', domain=mesh, subdomain_data=boundaries, subdomain_id=1)

#définition de la quantité d'intérêt
def J_2(w):
    return w*ds_sub


### 3e exemple : intégrale de la fonction sur un carré du maillage uniquement

mf = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
region = AutoSubDomain(lambda x, on: x[0] >= 0.25 and x[0] <= 0.5 and x[1] >= 0.25 and x[1]<= 0.5)
region.mark(mf, 1)
dx_sub = Measure('dx', subdomain_data=mf)

def J_3(w):
    return w*dx_sub(1)

############################


#définition de la tolérance d'erreur
tol = 1.e-5

#résolution du problème par la méthode adaptative, par rapport à u et aux conditions limites
problem = LinearVariationalProblem(a, L, u, bc) #définition du problème
solver = AdaptiveLinearVariationalSolver(problem, J_1(u)) #définition du solveur, en spécifiant la quantité d'intérêt J
solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "cg" # définition des paramètres
solver.solve(tol) #appel de la fonction pour la résolution, avec une tolerance donnée

solver.summary() #Affichage des résultats

# Récupération des graphes de la solution

#Sur le maillage initial
vtkfile = File("solution_primal_initial_mesh.pvd")
vtkfile << u.root_node()

#Sur le maillage final
vtkfile2 = File("solution_primal_final_mesh.pvd")
vtkfile2 << u.leaf_node()
