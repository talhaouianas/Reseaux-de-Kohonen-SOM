# -*- coding: utf-8 -*-
"""
Created on Monday Dec 21 22:28:11 2022

@author: anas
"""
import numpy as np  # destinée à manipuler des matrices ou tableaux

from matplotlib import pyplot as plt  # destinée à tracer et visualiser des données sous formes de graphiques


# Nous allons implémenter la SOM qui est utilisé pour la visualisation des données et qui fournit
# un résumé visuel rapide des instances de formation. En tant que grille 2D mxn, nécessitant un tableau NumPy 3D.
# Donc, La troisième dimension est requise pour stocker les poids dans chaque cellule.

# Etape 1 : Chercher le BMU
# : la fonction "cherche_BMU" permet de renvoi les coordonnées des cellules de la grille de la meilleure unité
# correspondante lorsqu'on lui donne la grille SOM et un exemple d'apprentissage x. Ensuite, il calcule le carré de
# la distance euclidienne entre chaque poids de cellule et x, puis il stock dans la variable distanceCarre et par
# conséquent, il renvoie (g,h), c'est-à-dire les coordonnées de la cellule avec la distance minimal(BMU)

def cherche_BMU(SOM, x):
    distanceCarre = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distanceCarre, axis=None), distanceCarre.shape)


# Etape 2 : Mettre à jour les poids des cellules SOM lorsqu'on leur donne un seul exemple d'entraînement
# et les paramètres du modèle avec les coordonnées BMU sous forme de tuple.


# La fonction mise_a_jour_weights() nécessite une grille SOM et un exemple d'apprentissage x, les paramètres taux_app
# et rayonCarre, les coordonnées de la meilleure unité correspondante et un paramètre step qui spécifie le nombre
# maximal des cellules à gauche, à droite, au-dessus et en dessous à modifier lors de la mise à jour du vecteur de poids
# Cependant, nous avons montré précédemment que le changement est négligeable pour les cellules éloignées de la BMU.
# Par conséquent, nous pouvons rendre le code plus efficace en ne changeant que les cellules dans un petit voisinage de la BMU.


def mise_a_jour_weights(SOM, exp_appx, taux_app, rayonCarre, BMU_coord, step=3):
    g, h = BMU_coord
    # si le rayon est proche de zéro alors seul BMU est modifié suivant la règle mentionner
    # 1e-3 notation exponentielle dans python est égale à 0.001
    if rayonCarre < 1e-3:
        SOM[g, h, :] += taux_app * (exp_appx - SOM[g, h, :])
        return SOM

        # Sinon il va Changer toutes les cellules dans un petit voisinage du BMU
    # (max(0,(g ou h)-step),min(SOM.shape[0],(g ou h)+step)
    # utiliser au cas où nous n'avons pas ces points dans la grille en respect les bordures de la grille

    for i in range(max(0, g - step), min(SOM.shape[0], g + step)):
        for j in range(max(0, h - step), min(SOM.shape[1], h + step)):
            distanceCarre = np.square(i - g) + np.square(j - h)  # Calcule de carre de la distance
            dist_func = np.exp(-distanceCarre / 2 / rayonCarre)  # calcule de la fonction distance
            SOM[i, j, :] += taux_app * dist_func * (exp_appx - SOM[i, j, :]) # modification de la pondération de la cellule encours
    return SOM


# Etape 3: Former la SOM

# La fonction forme_SOM() implémente la procédure d'apprentissage principale d'un SOM. Il nécessite une grille SOM
# initialisée ou partiellement formée et train_data comme paramètres. Le principe est de pouvoir former le SOM à
# partir d'un stade préalablement formé. En revanche, les paramètres learn_rate et radius_sq sont requis avec leurs
# taux de décroissance correspondants learn_decay et rayon_decay. Le paramètre epochs désigne le nombre d’itération
# d’apprentissage et qui est défini sur 10 par défaut mais peut être modifié si nécessaire.
# et par conséquent il renvoie la grille SOM.


def former_SOM(SOM, train_data, taux_app=.1, rayonCarre=1, learn_decay=.1, rayon_decay=.1, epochs=10):
    learn_rate_0 = taux_app
    rayon_0 = rayonCarre
    for epoch in np.arange(0, epochs):
        rand.shuffle(train_data)  # prendre aléatoirement une valeur
        for exp_appx in train_data:
            g, h = cherche_BMU(SOM, exp_appx)
            SOM = mise_a_jour_weights(SOM, exp_appx, taux_app, rayonCarre, (g, h))
        # Mettre à jour le taux d'apprentissage et le rayon
        taux_app = learn_rate_0 * np.exp(-epoch * learn_decay)
        rayonCarre = rayon_0 * np.exp(-epoch * rayon_decay)
    return SOM


# Dimensions de la grille SOM
n = 10
m = 10
# Nombre des exemples d'apprentissage
n_x = 3000
rand = np.random.RandomState(0)
# Initialize train_data
train_data = rand.randint(0, 255, (n_x, 3))
# Initialize SOM aléatoirement
SOM = rand.randint(0, 255, (m, n, 3)).astype(float)
# Display both the training matrix and the SOM grid

fig, ax = plt.subplots(
    nrows=3, ncols=3, figsize=(15, 15),
    subplot_kw=dict(xticks=[], yticks=[]))  # dictionnaire des clés
# Initialize the SOM randomly to the same state
for taux_app, i in zip([0.001, 0.5, 0.99], [0, 1, 2]):
    for rayonCarre, j in zip([0.01, 1, 10], [0, 1, 2]):
        rand = np.random.RandomState(0)
        SOM = rand.randint(0, 255, (m, n, 3)).astype(float)
        SOM = former_SOM(SOM, train_data, epochs=5, taux_app=taux_app,rayonCarre=rayonCarre)
        ax[i][j].imshow(SOM.astype(int))
        ax[i][j].title.set_text('$\eta$ = ' + str(taux_app) + ', $\sigma^2$ = ' + str(rayonCarre))

# Conclusion : un bon équilibre des paramètres est requis pour un entraînement idéal.
