"""
Module for data preparation for movies
The dataset used [1]_ is obtained from: https://grouplens.org/datasets/movielens/ and it has the following details:
Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
Link: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
More details for the dataset used can be found at: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html


"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

from typing import Tuple
import pandas as pd


def create_movie_users(ratings: pd.DataFrame,
                       movies: pd.DataFrame,
                       min_votes: int = 1) -> Tuple[pd.DataFrame,
                                                    pd.DataFrame,
                                                    dict,
                                                    dict]:
    """Data preparation for data movies. Creates user-movie matrix and
    maps movie ids to movie details.
    This function is based on the structure of MovieLens dataset [1]_

    Parameters
    ----------
    ratings : pd.DataFrame
        DataFrame of user ratings

    movies : pd.DataFrame
        DataFrame of movies

    min_votes : int
        Minimum number of total votes per user

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, dict, dict]
        user_movie_df : pd.DataFrame
            User - Movie matrix

        movie_ratings_df : pd.DataFrame
            User - Movie matrix with movie details

        movies_dict : dict
            Dictionary that maps movie ids to movie attributes

        id_titles_dict : dict
            Dictionary that maps movie ids to titles

    References
    ----------
    .. [1] F. Maxwell Harper and Joseph A. Konstan. 2015.
           The MovieLens Datasets: History and Context.
           ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872

    Examples
    --------
    >>> import pandas as pd
    >>> from ratingslib.app_movies.preparedata import create_movie_users
    >>> from ratingslib.datasets.filenames import MOVIES_SMALL_PATH, datasets_paths
    >>> MIN_VOTES = 200
    >>> filename_ratings, filename_movies = datasets_paths(MOVIES_SMALL_PATH+"ratings.csv", MOVIES_SMALL_PATH+"movies.csv")
    >>> ratings_df = pd.read_csv(filename_ratings)
    >>> movies_df = pd.read_csv(filename_movies)
    >>> user_movie_df, mr_df, movies_dict, id_titles_dict = create_movie_users(
        ratings_df, movies_df, min_votes=MIN_VOTES)
    """
    movies = movies[movies.title.notna()]
    movie_ratings_df = pd.merge(ratings, movies, on='movieId', how='inner')
    # movie_ratings = pd.merge(movies, ratings, on='movieId', how='inner')
    movie_ratings_df.groupby('movieId')[
        'rating'].count().reset_index(name="count")
    movie_ratings_df['count'] = movie_ratings_df.groupby(
        'movieId')["rating"].transform("count")
    movie_ratings_df = movie_ratings_df[movie_ratings_df['count'] >= min_votes]
    movie_ratings_df.sort_values(by='count', inplace=True, ascending=False)
    # non rated will be filled with 0
    user_movie_df = movie_ratings_df.pivot_table(
        index='userId', columns=['movieId'], values='rating').fillna(0)
    movies_dict = movies.set_index(movies.movieId).T.to_dict()
    id_titles_dict = {k: v['title'] for k, v in movies_dict.items()}
    return user_movie_df, movie_ratings_df, movies_dict, id_titles_dict
