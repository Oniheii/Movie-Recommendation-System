# Packages
import numpy as np
import pandas as pd
import random
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

# Fonction pour convertir le DataFrame en une matrice creuse CSR
# (une sous-matrice contenant uniquement les colonnes de notes)
def create_ratings_matrix(df_ratings):
    # Utilisation de la fonction pivot pour créer une matrice pivotée
    # avec les notes comme valeurs, les utilisateurs
    # comme index et les films comme colonnes
    pivot_table = df_ratings.pivot(
        index='userId', columns='movieId', values='rating')
    # Remplacement des valeurs manquantes par zéro
    pivot_table.fillna(0, inplace=True)
    # Conversion de la matrice pivotée en une
    # matrice creuse CSR (Compressed Sparse Row)
    ratings_matrix = csr_matrix(pivot_table)

    return ratings_matrix


# Fonction pour calculer la similarité Jaccard
def compute_jaccard_similarity(X):
    # Conversion de la matrice creuse en une matrice dense de type booléen
    dense_X = X.toarray().astype(bool)

    # Calcul de la similarité Jaccard en utilisant pairwise_distances
    # avec la métrique 'jaccard'
    jaccard_sim = 1 - pairwise_distances(dense_X, metric='jaccard')
    # Mettre la diagonale à 0 pour eviter la similarité avec soi-même
    np.fill_diagonal(jaccard_sim, 0)
    # arrondir les valeurs à retourner
    rounded_jaccard_sim = np.round(jaccard_sim, 2)

    return rounded_jaccard_sim


# Fonction pour calculer la similarité pearson
def compute_pearson_similarity(X):
    # Conversion de la matrice creuse CSR en une matrice dense et transposée
    dense_X = X.toarray().T
    # Calcul de la similarité Pearson uniquement pour les colonnes de notes
    # Calcul de la moyenne des notes par colonne (film)
    mean_ratings = np.nanmean(dense_X, axis=0)
    # Calcul de la différence entre les notes et leur moyenne respective
    X_centered = dense_X - mean_ratings
    # Calcul de la norme des notes centrées
    norm_ratings = np.sqrt(np.nansum(X_centered ** 2, axis=0))
    n_cols = dense_X.shape[1]
    pearson_sim = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            # Masque pour ne considérer que les valeurs non NaN
            mask = np.logical_not(
                np.logical_or(
                    np.isnan(dense_X[:, i]), np.isnan(dense_X[:, j])))
            if np.any(mask):
                # Calcul de la similarité Pearson en utilisant la formule
                similarity = np.sum(
                    X_centered[:, i][mask] * X_centered[:, j][mask]
                ) / (norm_ratings[i] * norm_ratings[j])
            else:
                similarity = 0.0
            pearson_sim[i, j] = similarity
            pearson_sim[j, i] = similarity
    # Mettre la diagonale à 0 pour eviter la similarité avec soi-même
    np.fill_diagonal(pearson_sim, 0)
    # arrondir les valeurs à retourner
    rounded_pearson_sim = np.round(pearson_sim, 2)

    return rounded_pearson_sim


# Fonction pour calculer la similarité cosine
def calculate_cosine_similarity(ratings_matrix):
    # Calculer la similarité cosinus en utilisant cosine_similarity
    cosine_sim = cosine_similarity(ratings_matrix)
    # Mettre la diagonale à 0 pour eviter la similarité avec soi-même
    np.fill_diagonal(cosine_sim, 0)
    # arrondir les valeurs à retourner
    rounded_cosine_sim = np.round(cosine_sim, 2)

    return rounded_cosine_sim


# Fonction pour calculer la similarité finale moyenne des trois
def compute_final_similarity(cosine_sim, jaccard_sim, pearson_sim):
    # Vérification des dimensions et formes des matrices
    assert (cosine_sim.shape == jaccard_sim.shape == pearson_sim.shape
            ), "Les dimensions des matrices de similarité doivent être identiques."
    # Calcul de la moyenne des similarités
    final_sim = (cosine_sim + jaccard_sim + pearson_sim) / 3
    # Mettre la diagonale à 0 pour eviter la similarité avec soi-même
    rounded_final_sim = np.round(final_sim, 2)

    return rounded_final_sim


# Fonction de normalisation des valeurs de matrices entre deux bornes données
def normalize_similarity(final_similarity, a=1, b=2):
    # Calcul des valeurs minimales et maximales de la matrice de similarité
    min_val = np.min(final_similarity)
    max_val = np.max(final_similarity)
    # Normalisation de la matrice de similarité en utilisant np.interp
    normalized_similarity = np.interp(
        final_similarity, (min_val, max_val), (a, b))
    print(f"La matrice de similarité finale (normalisée)"
          f" varie entre {a} et {b}.")

    return normalized_similarity
