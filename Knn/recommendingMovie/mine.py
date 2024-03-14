import data as da
import numpy as np
from sklearn.neighbors import NearestNeighbors



class knnMovieRecommendetion:

    def __init__(self, rate_matrix, user_ids, movie_ids, k = 5):
        self.k = k
        self.rate_m = rate_matrix
        self.user_ids = user_ids
        self.movie_ids = movie_ids


    def recommend(self, user_id, number):
        user_index = self.user_ids.index(user_id)
        user_ratings = self.rate_m[user_index]
        model =  NearestNeighbors(metric='cosine', algorithm='brute').fit(self.rate_m)
        distances, indices =model.kneighbors(self.rate_m, n_neighbors=self.k+1)
        similar_users_indices = indices.flatten()[1:]  # Exclude the user itself
        similar_users_ratings = self.rate_m[similar_users_indices]
        avg_ratings = np.mean(similar_users_ratings, axis=0)
        seen_movies_indices = np.where(user_ratings > 0)[0]
        for id in seen_movies_indices:
            avg_ratings[id] = 0

        recommended_movie_indices = np.argsort(-avg_ratings)[:number]
        recommended_movies = [self.movie_ids[i] for i in recommended_movie_indices]
        return recommended_movies






num_users = 200
num_movies = 100
synthetic_data, user_ids, movie_ids = da.generate_synthetic_data(num_users, num_movies)
ratings_matrix = np.array(synthetic_data)

knnModel = knnMovieRecommendetion(ratings_matrix , user_ids, movie_ids, 10)

recommended_movie_ids = knnModel.recommend('User_10', 5)
print("Recommended movies for User_1:", recommended_movie_ids)
