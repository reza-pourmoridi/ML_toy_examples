from sklearn.neighbors import NearestNeighbors
import numpy as np
import data as da

class MovieRecommender:
    def __init__(self, ratings_matrix, user_ids, movie_ids, k=5):
        self.ratings_matrix = ratings_matrix
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.k = k
        self.nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.nn_model.fit(ratings_matrix)

    def recommend_movies(self, user_id, num_recommendations=5):
        user_index = self.user_ids.index(user_id)
        user_ratings = self.ratings_matrix[user_index].reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(user_ratings, n_neighbors=self.k+1)
        similar_users_indices = indices.flatten()[1:]  # Exclude the user itself
        similar_users_ratings = self.ratings_matrix[similar_users_indices]

        avg_ratings = np.mean(similar_users_ratings, axis=0)
        unrated_movies_mask = user_ratings == 0
        # avg_ratings[~unrated_movies_mask] = 0  # Exclude rated movies
        recommended_movie_indices = np.argsort(-avg_ratings)[:num_recommendations]
        recommended_movies = [self.movie_ids[i] for i in recommended_movie_indices]
        return recommended_movies

# Example usage:

num_users = 200
num_movies = 100
synthetic_data, user_ids, movie_ids = da.generate_synthetic_data(num_users, num_movies)
ratings_matrix = np.array(synthetic_data)

recommender = MovieRecommender(ratings_matrix, user_ids, movie_ids , 10)
recommended_movies = recommender.recommend_movies('User_1')
print("Recommended movies for User_1:", recommended_movies)
