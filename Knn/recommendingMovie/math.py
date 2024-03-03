import numpy as np
import data as da  # Assuming you have a module named 'data' that provides the generate_synthetic_data function

class MovieRecommender:
    def __init__(self, ratings_matrix, user_ids, movie_ids, k=5):
        self.ratings_matrix = ratings_matrix
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.k = k

    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity

    def find_k_nearest_neighbors(self, user_ratings):
        similarities = []
        for other_user_ratings in self.ratings_matrix:
            similarity = self.cosine_similarity(user_ratings, other_user_ratings)
            similarities.append(similarity)
        similarities = np.array(similarities)
        nearest_neighbors_indices = np.argsort(similarities)[-self.k-1:-1]  # Get k nearest neighbors
        return nearest_neighbors_indices

    def recommend_movies(self, user_id, num_recommendations=5):
        user_index = self.user_ids.index(user_id)
        user_ratings = self.ratings_matrix[user_index]

        nearest_neighbors_indices = self.find_k_nearest_neighbors(user_ratings)

        avg_ratings = np.mean(self.ratings_matrix[nearest_neighbors_indices], axis=0)
        unrated_movies_mask = user_ratings == 0
        avg_ratings[~unrated_movies_mask] = 0  # Exclude rated movies

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
print(synthetic_data)
print("Recommended movies for User_1:", recommended_movies)
