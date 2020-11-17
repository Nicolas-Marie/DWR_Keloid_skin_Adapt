Optimisation du maillage du modèle de la cicatrice chéloïdienne suivant une loi d'élasticité linéaire par la méthode Dual-Weighted Residual (raffinement adaptatif)

Effectué avec FEniCS

Programme adapté du programme dwr-linear de Michel Duprez et Huu Phuoc Bui (2018) disponible a l'adresse :
https://figshare.com/articles/Quantifying_discretization_errors_for_softtissue_simulation_in_computer_assisted_surgery_a_preliminary_study/8128178
L'article associé y est egalement téléchargeable.

Système du modèle : - div(sigma(u)) = f  (f = 0 pour le modèle de chéloïde)

Conditions aux limites pré-définies :
Dirichlet + Neumann homogène

Le fichier du programme est : dwr_elasticite_lineaire.py

Les fichiers associés nécessaires pour l'exécution du code sont :

mesh_keloid_skin_pads_fixed.xml  (maillage initial)

mesh_keloid_skin_pads_fixed_facet_region.xml  (marqueurs zones chéloïde/peau saine pour les arêtes)

mesh_keloid_skin_pads_fixed_physical_region.xml  (marqueurs zones cheloide/peau saine pour les mailles)


Les résultats numériques en sortie sont enregistrés dans un fichier généré à l'exécution :
output_keloid_adapt_DWR.txt

Parmi ces résultats, on trouve l'estimateur global eta_h, l'estimateur local sum(eta_T) et l'erreur dans la quantite d'interet |J(u)-J(u_h)| pour chaque itération de l'algorithme de raffinement.

Deux quantités d'intérêt J sont déjà écrites en paramètre, elles peuvent être choisies tour à tour.
