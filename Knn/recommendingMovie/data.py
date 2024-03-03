import numpy as np

def generate_synthetic_data(num_users, num_movies, max_rating=5):
    """
    Generate synthetic movie rating data.

    Args:
    - num_users (int): Number of users in the dataset.
    - num_movies (int): Number of movies in the dataset.
    - max_rating (int): Maximum rating value.

    Returns:
    - ratings (numpy.ndarray): Array of shape (num_users, num_movies) containing synthetic ratings.
    - user_ids (list): List of user IDs.
    - movie_ids (list): List of movie IDs.
    """

    # Generate random ratings for each user-movie pair
    ratings = np.random.randint(0, max_rating+1, size=(num_users, num_movies))

    # Generate user IDs and movie IDs
    user_ids = ['User_' + str(i) for i in range(num_users)]
    movie_ids = ['Movie_' + str(i) for i in range(num_movies)]

    return ratings, user_ids, movie_ids

# Example usage:
# num_users = 1
# num_movies = 100
# synthetic_data, user_ids, movie_ids = generate_synthetic_data(1, 10)

