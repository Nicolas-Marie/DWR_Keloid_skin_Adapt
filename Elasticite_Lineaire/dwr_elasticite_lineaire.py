# Program's name : dwr_elasticite_lineaire
#
# Copyright (C) 2020 Nicolas Marie
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Optimisation du maillage du modele de la cicatrice cheloidienne
suivant une loi d'elasticite lineaire
par la methode Dual-Weighted Residual (raffinement adaptatif)


Auteurs : adapté du programme dwr-linear de Michel Duprez et Huu Phuoc Bui (2018) disponible a l'adresse :
https://figshare.com/articles/Quantifying_discretization_errors_for_softtissue_simulation_in_computer_assisted_surgery_a_preliminary_study/8128178
L'article associé y est egalement telechargeable.

Auteur de l'adaptation : Nicolas Marie (2020), pour l'application au cas de la cheloide
"""


"""
Systeme du modele : div(sigma(u)) = 0

Conditions limites : Dirichlet homogene (deplacement nul) au niveau du patin droit
                     Dirichlet avec deplacement u=(-0.3,0) au niveau du patin gauche
                     Neumann sur le reste de la frontiere

Fichiers necessaires pour l'execution du code :
mesh_keloid_skin_pads_fixed.xml (maillage initial)
mesh_keloid_skin_pads_fixed_facet_region.xml (marqueurs zones cheloide/peau saine pour les aretes)
mesh_keloid_skin_pads_fixed_physical_region.xml (marqueurs zones cheloide/peau saine pour les mailles)

Fichier de sortie des resultats numeriques : output_keloid_adapt_DWR.txt

Sorties : erreur explicite (explicit error)
          estimateur global eta_h
          estimateur local sum(eta_T)
          erreur dans la quantite d'interet (interest error) |J(u)-J(u_h)|
          quantite d'interet "exacte" J(u) (calculee sur maillage tres fin)
          erreur relative de la quantite d'interet |J(u)-J(u_h)|/|J(u)|
          norme L2 de l'erreur dans le calcul du deplacement u
          indice d'efficacite de l'estimateur global : eta_h / |J(u)-J(u_h)|
          indice d'efficacite de l'estimateur local : sum(eta_T) / |J(u)-J(u_h)|
"""

from dolfin import *

import numpy as np
import os
import os.path as osp
import time

parameters["allow_extrapolation"] = True # pour le raffinement du maillage
parameters["refinement_algorithm"] = "plaza_with_parent_facets" # pour le raffinement du maillage


###################################
#####   Donnees du probleme   #####
###################################

# Modules de Young
module_young_h = 0.01 # partie peau saine
ratio = 50 # ratio E_k/E_h
E_h = Constant(module_young_h) # partie peau saine
E_k = Constant(ratio*module_young_h) # partie cheloide

# Coefficient de Poisson (identique pour les deux zones)
nu_k = Constant(0.49)
nu_h = Constant(0.49)

# Force externe
f = Constant((0,0))

#Conditions aux limites
u_D1 = Constant((-0.3, 0.0)) # Dirichlet sur le patin gauche
u_D2 = Constant((0.0, 0.0)) # Dirichlet sur le patin droit (deplacement nul)


#### Quantite d'interet ####

# Intégrale de la trace de sigma
def J(u,dX,Lmbda,Mu): # dX mesure pour calculs d'intégrales, Lmbda et Mu coefficients de Lame adaptes au maillage souhaite
    return tr(sigma(u,Lmbda,Mu))*dX(10) + tr(sigma(u,Lmbda,Mu))*dX(20)
    #integration sur la zone des observations (partie cheloide + partie peau saine avec mesures)
    #region marquee par 10 pour la cheloide, par 20 pour la peau saine avec mesures

# # Intégrale du cisaillement (sigma_xy)
# def J(u,dX,Lmbda,Mu):
#     return sigma(u,Lmbda,Mu)[0,1]*dX(10) + sigma(u,Lmbda,Mu)[0,1]*dX(20)


# Degre des elements finis
degreElemts = 2

# Critere d'arret de l'algorithme adaptatif (defini par rapport a l'estimateur global eta_h)
tol = 5E-4

# taille maximale des tableaux en sortie
taille_max = 10

# Nombre de raffinements uniformes a effectuer pour obtenir le maillage de la solution "exacte"
iter_ref_exact = 1

# Coefficient de raffinement dans le marquage de Dorfler
refinement_fraction = 0.8

# Parametres pour les sorties
Plot = True

if Plot == True: # import des modules graphiques si Plot est "True"
    import matplotlib.pyplot as plt
    import matplotlib as mpl

Save = True #pour la sauvegarde de certaines figures

#############################
##### fin des donnees    ####
#############################



####################################
##### Definition des fonctions #####
####################################

# Estimateur residuel explicite (voir Babuska et al 1978, ou Verfurth 1999 (p.424))
# Renvoie un vecteur
def explicit_residual(u_h,DG0,h,n):
        w = TestFunction(DG0) #fonctions constantes egales a 1 sur un element et 0 partout ailleurs

        # Residu a l'interieur du domaine
        residual1 = h**2*w*(div(sigma(u_h,lmbda,mu))+f)**2*dx

        # Saut aux facettes interieures du domaine
        residual2 = avg(w)*avg(h)*inner(sigma(u_h,lmbda,mu)('+')*n('+')+sigma(u_h,lmbda,mu)('-')*n('-'),sigma(u_h,lmbda,mu)('+')*n('+')+sigma(u_h,lmbda,mu)('-')*n('-'))*dS
         # dS: facettes interieures, avg(w) = (w('+') + w('-'))/2

        # Saut a la frontiere de Neumann
        residual3 = w*h*(-sigma(u_h,lmbda,mu)*n)**2*ds(0) # ds(0) : frontiere de Neumann

        #Regroupement des termes
        residual =  residual1 + residual2 + residual3
        cell_residual = Function(DG0)
        error_explicite = assemble(residual, tensor=cell_residual.vector()).get_local() #vecteur
        return error_explicite


# Fonction indiquant les elements du maillage a raffiner lors de l'execution de l'algorithme adaptatif.
# Raffinement a partir de l'indicateur eta (vecteur des indicateurs eta_T pour chaque element)
def dorfler_marking(eta, mesh):
        # eta_ind de taille 2xn, 1ere ligne pour eta et 2eme pour les indices
        eta_ind = np.concatenate((np.array([eta]),np.array([range(len(eta))])),axis=0) #Concatene les 2 tableaux le long de l'axe des lignes
        eta_ind = eta_ind.T[np.lexsort(np.fliplr(eta_ind.T).T)].T #classe les indicateurs par ordre croissant ; fliplr echange les colonnes
        eta_ind = eta_ind[:,::-1] # met les indicateurs en ordre decroissant

        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)

        ind=0
        while np.sum(eta_ind[0,:ind]) < refinement_fraction*np.sum(eta):
                cell_markers[int(eta_ind[1,ind])] = True
                ind=ind+1

        return cell_markers


# Calcul de la solution sur un maillage tres fin, consideree comme la solution "exacte"
def func_u_exact(mesh,materials,edges):
        mesh_exact = Mesh(mesh) #recuperation du maillage passé en parametre
        materials_exact = MeshFunction("size_t",mesh_exact, mesh.topology().dim()) # MeshFunction pour les elements du maillage exact
        edges_exact = MeshFunction("size_t",mesh_exact, mesh.topology().dim() - 1) # MeshFunction pour les aretes du maillage exact

        # Copie des MeshFunction en parametres dans les MeshFunction qui serviront au maillage de la solution "exacte"
        for i in range(mesh_exact.num_cells()):
                materials_exact[i] = materials[i]

        for i in range(mesh_exact.num_cells()):
            for fct in facets(Cell(mesh_exact,i)):
                edges_exact[fct] = edges[fct]

        # Idem, copie de l'espace de fonction DGO2
        DG02_exact = DG02

        # Raffinement uniforme (1 ou plusieurs) du maillage et adaptation des MeshFunction et de l'espace de fonctions au nouveau maillage
        ### Remarque : le maillage etant issu du maillage raffine adaptativement (voir ligne 1 de cette fonction),
        ### il est encore plus fin a chaque iteration, et les mailles sont plus fines
        ### aux memes endroits que pour le maillage raffine adaptativement
        for i in range(iter_ref_exact):
                mesh_exact_refined = refine(mesh_exact)
                materials_exact = adapt(materials_exact, mesh_exact_refined)
                edges_exact = adapt(edges_exact, mesh_exact_refined)
                DG02_exact = VectorFunctionSpace(mesh_exact_refined,'DG',0,2) # DG pour "Discontinuous Galerkin", degre 0 (constant par element), 2 dimensions
                mesh_exact = mesh_exact_refined

        print("Nombre d elements pour la solution exacte : ", mesh_exact.num_cells())

        #Espace d'elements finis vectoriels sur le maillage exact
        V_exact = VectorFunctionSpace(mesh_exact,'CG',degreElemts,2) # CG pour Continuous Garlerkin (i.e. elements de Lagrange), avec dimension=2

        # Initialise les mesures pour les calculs d'integrales par rapport aux domaines que l'on vient de definir
        dx_exact = Measure("dx")(subdomain_data = materials_exact)
        ds_exact = Measure("ds")(subdomain_data = edges_exact) # facettes exterieures

        # Prise en compte des parametres E et nu qui varient en fonction du materiau, sous forme d'expressions
        class E_expression_exacte(UserExpression):
            def __init__(self, materials_exact, **kwargs):
                self.materials_exact = materials_exact
                super().__init__(**kwargs)

            def eval_cell(self, values, x, cell):
                if self.materials_exact[cell.index] == 10: # partie cheloide
                    values[0] = E_k
                elif self.materials_exact[cell.index] == 20: # partie peau saine avec mesures
                    values[0] = E_h
                elif self.materials_exact[cell.index] == 30: # partie peau saine sans mesures
                    values[0] = E_h

            def value_shape(self):
                return ()


        class nu_expression_exacte(UserExpression):
            def __init__(self, materials_exact, **kwargs):
                self.materials_exact = materials_exact
                super().__init__(**kwargs)

            def eval_cell(self, values, x, cell):
                if self.materials_exact[cell.index] == 10: # partie cheloide
                    values[0] = nu_k
                elif self.materials_exact[cell.index] == 20: # partie peau saine avec mesures
                    values[0] = nu_h
                elif self.materials_exact[cell.index] == 30: # partie peau saine sans mesures
                    values[0] = nu_h

            def value_shape(self):
                return ()


        E_exact = E_expression_exacte(materials_exact, degree=0)
        nu_exact = nu_expression_exacte(materials_exact, degree=0)

        # Coefficients de Lame
        lmbda_exact = E_exact*nu_exact/((1-2*nu_exact)*(1+nu_exact))
        mu_exact = E_exact/(2*(1+nu_exact))


        # Conditions de Dirichlet
        bc_pad_one_exact = DirichletBC(V_exact, u_D1, edges_exact, 1) # patin gauche sans le "sensor"
        bc_pad_one_sensor_exact = DirichletBC(V_exact, u_D1, edges_exact, 3) # sensor du patin gauche
        bc_pad_two_exact = DirichletBC(V_exact, u_D2, edges_exact, 2) # patin droit
        bcs_exact = [bc_pad_one_exact, bc_pad_one_sensor_exact, bc_pad_two_exact]

        # Definition du probleme variationnel
        u_trial_exact = TrialFunction(V_exact)
        v_exact = TestFunction(V_exact)
        a_exact = inner(sigma(u_trial_exact,lmbda_exact,mu_exact),epsilon(v_exact))*dx_exact
        L_exact = inner(f,v_exact)*dx_exact

        # Solution calculee
        u_exact = Function(V_exact)

        # Resolution de l'equation a_exact = L_exact par rapport a u_exact
        problem = LinearVariationalProblem(a_exact, L_exact, u_exact, bcs_exact)
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "mumps"
        solver.solve()

        return  mesh_exact, V_exact, dx_exact, u_exact, lmbda_exact, mu_exact


# Tenseur des deformations
def epsilon(u):
    return sym(grad(u))

# Tenseur des contraintes
def sigma(u,lmbda,mu): # lmbda et mu coefficients de Lame mis en parametres car actualisés a chaque raffinement du maillage
    return lmbda*tr(sym(grad(u)))*Identity(2)+2.0*mu*sym(grad(u))


# Fonction utilisee pour ecrire dans les fichiers en sortie
def output_latex(fl,A,B):
        for i in range(len(A)):
                fl.write('(')
                fl.write(str(A[i]))
                fl.write(', ')
                fl.write(str(B[i]))
                fl.write(')\n')
        fl.write('\n')

########################################
##### Fin Definition des fonctions #####
########################################


# Initialisation des tableaux pour les sorties
# Vecteurs ligne a taille_max elements
num_cell_array = np.zeros(taille_max)
explicit_array = np.zeros(taille_max)
sum_eta_T_array = np.zeros(taille_max)
eta_h_array = np.zeros(taille_max)
interest_error_array = np.zeros(taille_max)
exact_interest_array = np.zeros(taille_max)
approx_interest_array = np.zeros(taille_max)
interest_relative_error_array = np.zeros(taille_max)
global_error_array = np.zeros(taille_max)
eta_h_effi_array = np.zeros(taille_max)
sum_eta_T_effi_array = np.zeros(taille_max)


# Generation du maillage initial
mesh = Mesh("mesh_keloid_skin_pads_fixed.xml")


# MeshFunction pour marquer les zones cheloide et peau saine
# Vaut 10 pour la zone de la cheloide, 20 pour la peau saine ou des observations sont faites,
# et 30 pour la peau saine ou aucune observation n'est faite
materials = MeshFunction("size_t", mesh, mesh.topology().dim())
#Recuperation dans la MeshFunction du fichier ou se trouvent les marqueurs cheloide/peau saine
File('mesh_keloid_skin_pads_fixed_physical_region.xml') >> materials

# MeshFunction pour marquer les aretes
edges = MeshFunction("size_t",mesh, mesh.topology().dim()-1)
#Recuperation dans la MeshFunction du fichier ou se trouvent les marqueurs cheloide/peau saine
File("mesh_keloid_skin_pads_fixed_facet_region.xml") >> edges

#Definition d'espaces de fonctions constantes sur chaque element
DG02 = VectorFunctionSpace(mesh,'DG',0,2) # "Discontinuous Galerkin", degre 0 (constant par element), 2 dimensions
DG0 = FunctionSpace(mesh,'DG',0)



############################################
######      Algorithme adaptatif       #####
############################################

start_time = time.time() # pour calculer le temps d'exécution de l'algorithme

init = True # booleen indiquant s'il s'agit de la premiere iteration ou non
ite = 0 # compteur indiquant l'iteration en cours

while ite==0 or eta_h > tol :
# Boucle avec critere d'arret defini par rapport a l'estimateur global eta_h
        print("##############################")
        print("ITERATION:",ite+1)
        if init == True: # on ne raffine pas a la premiere iteration
                init = False
        else: #raffinement adaptatif du maillage et adaptation des MeshFunction et de l'espace de fonction
                mesh_refined = refine(mesh, cell_markers) #raffine le maillage localement ou cell_markers l'indique
                materials = adapt(materials, mesh_refined)
                edges = adapt(edges, mesh_refined)
                DG02 = VectorFunctionSpace(mesh_refined,'DG',0,2)
                mesh = mesh_refined

        if Plot == True:
            ## Sauvegarde du maillage a chaque iteration ##

            plt.figure()
            plot_mesh = plot(mesh, title="Maillage a l\'iteration {}".format(ite+1))
            # Suppression des anciennes figures au cas ou le script aurait deja ete execute
            if ite == 0 and osp.exists("Meshes/eps"):
                for file in os.listdir("Meshes/eps"):
                    os.remove("Meshes/eps/{}".format(file))
            # (Re)creation du dossier contenant les figures
            if not osp.exists("Meshes/eps"):
                os.makedirs("Meshes/eps")
            # Idem pour le format png
            if ite == 0 and osp.exists("Meshes/png"):
                for file in os.listdir("Meshes/png"):
                    os.remove("Meshes/png/{}".format(file))
            if not osp.exists("Meshes/png"):
                os.makedirs("Meshes/png")

            plt.savefig('Meshes/eps/mesh_ite_{}.eps'.format(ite+1))
            plt.savefig('Meshes/png/mesh_ite_{}.png'.format(ite+1))


        # Definition du vecteur normal pour chaque facette et du diametre pour chaque element
        n = FacetNormal(mesh)
        h = CellDiameter(mesh)

        print("Nombre d elements : ", mesh.num_cells())

        # Construction des differents espaces
        V = VectorFunctionSpace(mesh,'CG',degreElemts,2) # 2 pour la dimension
        DG0 = FunctionSpace(mesh,'DG',0)

        # Espace pour la solution duale
        V_star2 = VectorFunctionSpace(mesh, "CG",degreElemts+1, 2)

        ## Initialise les mesures pour integrer sur les nouveaux domaines, dS (sur les facettes interieures), ds (facettes exterieures)
        dx = Measure("dx")(subdomain_data = materials)
        dS = Measure("dS")(subdomain_data = materials)
        ds = Measure("ds")(subdomain_data = edges)


        class E_expression(UserExpression):
            def __init__(self, materials, **kwargs):
                self.materials = materials
                super().__init__(**kwargs)

            def eval_cell(self, values, x, cell):
                if self.materials[cell.index] == 10: # partie cheloide
                    values[0] = E_k
                elif self.materials[cell.index] == 20: # peau saine mesuree
                    values[0] = E_h
                elif self.materials[cell.index] == 30: # peau saine non mesuree
                    values[0] = E_h

            def value_shape(self):
                return ()


        class nu_expression(UserExpression):
            def __init__(self, materials, **kwargs):
                self.materials = materials
                super().__init__(**kwargs)

            def eval_cell(self, values, x, cell):
                if self.materials[cell.index] == 10: # partie cheloide
                    values[0] = nu_k
                elif self.materials[cell.index] == 20: # peau saine mesuree
                    values[0] = nu_h
                elif self.materials[cell.index] == 30: # peau saine non mesuree
                    values[0] = nu_h

            def value_shape(self):
                return ()


        E = E_expression(materials, degree=0)
        nu = nu_expression(materials, degree=0)

        # Coefficients de Lame
        lmbda = E*nu/((1-2*nu)*(1+nu))
        mu = E/(2*(1+nu))


        # Conditions de Dirichlet
        bc_pad_one = DirichletBC(V, u_D1, edges,1) # patin gauche sans le "sensor"
        bc_pad_one_sensor = DirichletBC(V, u_D1, edges, 3) # sensor du patin gauche
        bc_pad_two = DirichletBC(V, u_D2, edges,2) # patin droit
        bcs = [bc_pad_one,bc_pad_one_sensor, bc_pad_two]


        ##---------------------------------------- Probleme primal
        # Definition du probleme variationnel
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(sigma(u,lmbda,mu),epsilon(v))*dx
        L = inner(f,v)*dx

        # Calcul de la solution
        u_h = Function(V)
        # Resolution de l'equation a = L par rapport a u et les conditions limites
        problem = LinearVariationalProblem(a, L, u_h, bcs)
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "mumps"
        solver.solve()

        #Tracé des graphes de la solution sur le maillage grossier de depart
        if Plot == True:
            if ite==0:
                plt.figure()
                plot_u_h_x_init = plot(u_h[0],title = 'Premiere composante de u_h sur le maillage grossier')
                plt.savefig('first_component_u_h_init.png')

                plt.figure()
                plot_u_h_y_init = plot(u_h[1],title = 'Deuxieme composante de u_h sur le maillage grossier')
                plt.savefig('second_component_u_h_init.png')

                File_u_h_init = File("solution_u_h_init.pvd")
                File_u_h_init << u_h


        # 1) ---------------------------------------- Calcul de l'erreur explicite (Babuska et al 1978, Verfurth 1999 (p.424))
        error_explicite = explicit_residual(u_h,DG0,h,n)
        print('Erreur residuelle explicite : ', sum(abs(error_explicite))**(0.5))


        ##------------------------------------------ Probleme dual
        # Definition du probleme dual variationnel dans l'ensemble des polynomes P(k+1), (degre superieur de 1 par rapport au primal)
        u_star2 = TrialFunction(V_star2)
        v_star2 = TestFunction(V_star2)
        a_star2 = inner(sigma(v_star2,lmbda,mu),epsilon(u_star2))*dx
        L_star2 = J(v_star2,dx,lmbda,mu)


        # Conditions de Dirichlet (toutes homogenes pour la solution duale)
        bc_pad_one = DirichletBC(V_star2, Constant((0,0)), edges,1) # patin gauche sans le "sensor"
        bc_pad_one_sensor = DirichletBC(V_star2, Constant((0,0)), edges, 3) # sensor du patin gauche
        bc_pad_two = DirichletBC(V_star2, Constant((0,0)), edges,2) # patin droit
        bcs = [bc_pad_one,bc_pad_one_sensor, bc_pad_two]

        z_h = Function(V_star2)
        problem = LinearVariationalProblem(a_star2, L_star2, z_h, bcs)
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "mumps"
        solver.solve()


        # 2)------------------ Calcul de l'estimateur global eta_h
        # On a J(u)-J(u_h) =  r(z) = l(z) - a(u_h,z), ainsi que eta_h = abs(r(z_h))
        sigma_u_h = sigma(u_h,lmbda,mu)
        rEz_h = assemble(inner(f,z_h)*dx - inner(sigma_u_h,epsilon(z_h))*dx)
        eta_h = abs(rEz_h)
        print("eta_h : ",eta_h)


        # 3)----------------- Calcul de l'estimateur local sum(eta_T)
        diff = z_h - interpolate(z_h,V) # z_h est dans V_star2
        w = TestFunction(DG0)

        eta_T1 = w*inner(f + div(sigma_u_h),diff)*dx # Residu interieur
        eta_T2 = -2.0*avg(w)*0.5*inner(sigma_u_h('+')*n('+')+sigma_u_h('-')*n('-'),avg(diff))*dS # Saut aux facettes interieures
        eta_T3 = w*inner(- sigma_u_h*n,diff)*ds(0) # Saut a la frontiere de Neumann (marquee par 0)

        eta_T = eta_T1 + eta_T2 + eta_T3

        eta = abs(assemble(eta_T).get_local()) # assemble pour chaque element T (eta est un vecteur)

        sum_eta_T = sum(eta) #calcul de la somme de tous les indicateurs locaux pour obtenir l'estimateur
        print("Sum(eta_T) : ", sum_eta_T)


        # Cartographie des indicateurs locaux eta_T à chaque itération (au format pvd)
        eta_T_map = MeshFunction("double", mesh, mesh.topology().dim())
        for i in range(len(eta)):
                eta_T_map[i] = eta[i]

        File_eta_T_map = File("eta_T_map_ite_{}.pvd".format(ite+1))
        File_eta_T_map << eta_T_map


        #####----------- CALCULS

        # Calcul de la solution "exacte"
        mesh_exact, V_exact, dx_exact, u_exact, lmbda_exact, mu_exact = func_u_exact(mesh,materials,edges)
        # La fonction permet de recuperer aussi le maillage "exact", l'espace de fonction et la mesure correspondants

        # # Norme L2 de u_h
        # print('Norme L2 de u_h : ', assemble(inner(u_h,u_h)*dx)**(0.5))

        # ### Affichage et sauvegarde du maillage pour la solution exacte ###
        # if Plot == True:
        #     plt.figure()
        #     plot_mesh_exact = plot(mesh_exact, title="Maillage pour la solution exacte a l\'iteration {}".format(ite))

        # interpolation de u_h a l'espace V_exact (avec un maillage tres fin)
        Iu_h = interpolate(u_h,V_exact)

        # # Norme de la solution exacte
        # print('Norme de la solution exacte : ',assemble(inner(u_exact,u_exact)*dx_exact)**(0.5))

        # # Norme L_2 globale de l'erreur dans le calcul du deplacement u
        # global_error = abs(assemble(inner(u_exact-Iu_h,u_exact-Iu_h)*dx_exact)**(0.5))
        # print('Norme L2 de l erreur dans le calcul du deplacement u : ', global_error)

        # Erreur dans la quantite d'interet
        interest_error = abs(assemble(J(u_exact,dx_exact,lmbda_exact,mu_exact)) - assemble(J(u_h,dx,lmbda,mu)))
        print('Erreur dans la quantite d interet |J(u)-J(u_h)| : ', interest_error)

        # Valeur de la quantite d'interet approchee
        approx_interest = assemble(J(u_h,dx,lmbda,mu))
        print("Quantite d interet approchee J(u_h) : ",approx_interest)

        # Quantite d'interet "exacte"
        exact_interest = assemble(J(u_exact,dx_exact,lmbda_exact,mu_exact))
        print('Quantite d interet exacte J(u) : ',exact_interest)

        # Erreur relative de la quantite d'interet
        print('Erreur relative de la quantite d interet |J(u)-J(u_h)|/|J(u)| : ', interest_error/abs(exact_interest))

        # Indice d'efficacite de l'estimateur global
        print('Eta_h / |J(u)-J(u_h)| : ', eta_h/interest_error)

        # Indice d'efficacite de l'estimateur local
        print('Sum(eta_T) / |J(u)-J(u_h)| : ', sum_eta_T/interest_error)

        # Graphes champs de contrainte et trace de sigma
        if Plot == True:
            plt.figure()
            plot_sigma_xx = plot(sigma(u_h,lmbda,mu)[0,0], title = 'Contrainte sigma_xx a l\'iteration {}'.format(ite+1))
            plt.savefig('sigma_xx_ite_{}.png'.format(ite+1))

            plt.figure()
            plot_sigma_yy = plot(sigma(u_h,lmbda,mu)[1,1], title = 'Contrainte sigma_yy a l\'iteration {}'.format(ite+1))
            plt.savefig('sigma_yy_ite_{}.png'.format(ite+1))

            plt.figure()
            plot_sigma_xy = plot(sigma(u_h,lmbda,mu)[0,1], title = 'Contrainte sigma_xy a l\'iteration {}'.format(ite+1))
            plt.savefig('sigma_xy_ite_{}.png'.format(ite+1))

            plt.figure()
            plot_tr_sigma = plot(tr(sigma(u_h,lmbda,mu)), title = 'tr(sigma) a l\'iteration {}'.format(ite+1))
            plt.savefig('tr_sigma_ite_{}.png'.format(ite+1))

        ### au format pvd :
        V_tensor = TensorFunctionSpace(mesh,'CG',degreElemts)
        sigm = project(sigma(u_h,lmbda,mu),V_tensor)
        File_sigma = File("sigma_ite_{}.pvd".format(ite+1))
        File_sigma << sigm
        # trace de sigma
        V_scalar = FunctionSpace(mesh,'CG',degreElemts)
        tr_sigm = project(tr(sigma(u_h,lmbda,mu)),V_scalar)
        File_tr_sigma = File("tr_sigma_ite_{}.pvd".format(ite+1))
        File_tr_sigma << tr_sigm


        # Construction de l'indicateur pour le raffinement
        cell_markers = dorfler_marking(eta, mesh)

        # Mise a jour des tableaux
        num_cell_array[ite] = mesh.num_cells()
        explicit_array[ite] = sum(abs(error_explicite))**(0.5)
        sum_eta_T_array[ite] = sum_eta_T
        eta_h_array[ite] = eta_h
        interest_error_array[ite] = interest_error
        exact_interest_array[ite] = exact_interest
        approx_interest_array[ite] = approx_interest
        interest_relative_error_array[ite] = interest_error/abs(exact_interest)
        # global_error_array[ite] = global_error
        eta_h_effi_array[ite] = eta_h/interest_error
        sum_eta_T_effi_array[ite] = sum_eta_T/interest_error

        ite = ite + 1

################################################
############# Fin de l'algorithme ##############
################################################

# Calcul du temps de calcul de l'algorithme
interval = time.time() - start_time
print('Temps de calcul de l algorithme adaptatif en secondes : ', interval)

#### Ecriture des differents tableaux dans les fichiers en sortie ####
fl = open('output_keloid_adapt_DWR.txt','w')

fl.write('Ratio de raffinement du marquage de Dorfler : ')
fl.write(str(refinement_fraction))
fl.write('\n')

fl.write('Nombre d iterations pour le raffinement du maillage de la solution exacte : ')
fl.write(str(iter_ref_exact))
fl.write('\n')

fl.write('Degre des polynomes de l\'espace d\'elements finis : ')
fl.write(str(degreElemts))
fl.write('\n')

fl.write('\n')
fl.write('Nombre d\'elements et Erreur explicite a chaque iteration : \n')
output_latex(fl,num_cell_array,explicit_array)
fl.write('Nombre d\'elements et Eta_h a chaque iteration : \n')
output_latex(fl,num_cell_array,eta_h_array)
fl.write('Nombre d\'elements et sum(eta_T) a chaque iteration : \n')
output_latex(fl,num_cell_array,sum_eta_T_array)
fl.write('Erreur de la quantite d\'interet |J(u)-J(u_h)| : \n')
output_latex(fl,num_cell_array,interest_error_array)
fl.write('Quantite d\'interet exacte J(u) : \n')
output_latex(fl,num_cell_array,exact_interest_array)
fl.write("Quantite d\'interet approchee J(u_h) : \n")
output_latex(fl,num_cell_array,approx_interest_array)
fl.write('Erreur relative de la quantite d\'interet |J(u)-J(u_h)|/|J(u)| : \n')
output_latex(fl,num_cell_array,interest_relative_error_array)
# fl.write('Erreur en norme L2 dans le calcul du deplacement u : \n')
# output_latex(fl,num_cell_array,global_error_array)
fl.write('Eta_h / |J(u)-J(u_h)| : \n')
output_latex(fl,num_cell_array,eta_h_effi_array)
fl.write('Sum eta_T / |J(u)-J(u_h)| : \n')
output_latex(fl,num_cell_array,sum_eta_T_effi_array)
fl.close()



# Tracé des graphes
if Plot == True:
        plt.figure()
        plot_mesh_final = plot(mesh,title='Maillage final')
        plt.savefig('final_mesh.eps')
        plt.savefig('final_mesh.png')

        plt.figure()
        plot_u_h_x = plot(u_h[0],title = 'Premiere composante de u_h')
        plt.savefig('first_component_u_h.png')

        plt.figure()
        plot_u_h_y = plot(u_h[1],title = 'Deuxieme composante de u_h')
        plt.savefig('second_component_u_h.png')

        plt.figure()
        plot_u_h_norm = plot((abs(u_h[0])**2+abs(u_h[1])**2)**0.5,title='Norme de u_h(x,y)')
        plt.savefig('Norm_u_h.png')

        plt.figure()
        plot_z_h_x = plot(z_h[0],title = 'Premiere composante de z_h')
        plt.savefig('first_component_z_h.png')

        plt.figure()
        plot_z_h_y = plot(z_h[1],title = 'Deuxieme composante de z_h')
        plt.savefig('second_component_z_h.png')

        plt.figure()
        plot_z_h_norm = plot((abs(z_h[0])**2+abs(z_h[1])**2)**0.5,title='Norme de z_h(x,y)')
        plt.savefig('Norm_z_h.png')

        #### Courbes des resultats numeriques ####
        plt.figure()
        mpl.rc('text', usetex=True) #pour ecrire en latex dans la legende
        plt.plot(num_cell_array[0:ite],eta_h_array[0:ite],'+',linestyle='-',label=r'$\eta_h$')
        plt.plot(num_cell_array[0:ite],sum_eta_T_array[0:ite],'+',linestyle='-',label=r'$\sum \eta_T$')
        plt.plot(num_cell_array[0:ite],interest_error_array[0:ite],'+',linestyle='-',label=r'$|J(u)-J(u_h)|$')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Nombre d elements du maillage')
        plt.savefig("courbe_d_erreur.eps")
        plt.savefig("courbe_d_erreur.png")


# Sauvegardes de certaines figures au format pvd
if Save == True:
        File_final_mesh = File("final_mesh.pvd")
        File_final_mesh << mesh
        File_u_h = File("solution_u_h.pvd")
        File_u_h << u_h
        File_z_h = File("solution_z_h.pvd")
        File_z_h << z_h
