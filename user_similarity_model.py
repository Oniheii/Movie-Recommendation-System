# Packages
import numpy as np
import pandas as pd
import random


# fonction pour selectionner aléatoire N valeurs à supprimer pour prédir
def select_random_votes(df_ratings, n):
    # Sélectionne aléatoirement n votes à supprimer pour la prédiction.
    random_votes = df_ratings.sample(n, random_state=42)
    ratings_for_prediction = df_ratings.drop(random_votes.index)
    return random_votes, ratings_for_prediction


# fonction pour prédictir les valeurs d'évaluation
def predict_ratings(final_similarity, ratings_for_prediction, random_votes):
    # Prédit les évaluations manquantes en utilisant la matrice de similarité.

    # Dictionnaire pour stocker les évaluations par utilisateur et film
    ratings_dict = {}
    for index, row in ratings_for_prediction.iterrows():
        user = int(row['userId'])
        item = row['movieId']
        rating = row['rating']

        if user not in ratings_dict:
            ratings_dict[user] = {}

        ratings_dict[user][item] = rating

    predicted_ratings = []

    for index, row in random_votes.iterrows():
        user = int(row['userId'])
        item = row['movieId']
        similarity_sum = 0
        weighted_sum = 0

        for other_user in range(1, final_similarity.shape[1] + 1):
            if other_user != user:
                similarity = final_similarity[user - 1, other_user - 1]

                if item in ratings_dict[other_user]:
                    rating = ratings_dict[other_user][item]
                    similarity_sum += similarity
                    weighted_sum += similarity * rating

        prediction =\
        weighted_sum / similarity_sum if similarity_sum != 0 else 0
        predicted_ratings.append(prediction)
    # prédiction dans une table de données
    df_predicted_ratings = random_votes.copy()
    df_predicted_ratings['rating'] = predicted_ratings
    df_predicted_ratings.rename(
        columns={'rating': 'rating_predicted'}, inplace=True)

    return df_predicted_ratings


# Fonction pour calculer le MAE
def calculate_mae(predictions, actual_ratings):
    # Calcul de l'erreur absolue entre les prédictions et les évaluations réelles
    errors = np.abs(predictions['rating_predicted'] - actual_ratings['rating'])
    # Calcul du MAE en prenant la moyenne des erreurs
    mae = np.mean(errors)
    # Arrondir le résultat à deux décimales
    rounded_mae = np.round(mae, 2)

    return rounded_mae


# Fonction pour calculer le Rappel (recall)
def calculate_recall(predictions, actual_ratings, k):
    # Tri des prédictions par ordre décroissant
    sorted_predictions = predictions.sort_values('rating_predicted', ascending=False)
    # Obtention des k premières prédictions triées
    top_k_userIds = sorted_predictions['userId'].head(k)
    # Ensemble des éléments pertinents
    denominator = actual_ratings[actual_ratings['rating'] >= 4]['rating'].sum()
    # Ensemble des éléments pertinents dans le top-k
    relevant_ratings = actual_ratings[
        (actual_ratings['userId'].isin(top_k_userIds)
         ) & (actual_ratings['rating'] >= 4)]
    numerator = relevant_ratings['rating'].sum()
    # Calcul du recall
    if denominator > 0 and numerator > 0:
        recall = numerator / denominator
    else:
        recall = 0
    # arrondir les valeurs à retourner
    rounded_recall = np.round(recall, 2)

    return rounded_recall
