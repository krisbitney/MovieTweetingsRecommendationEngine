import numpy as np
import pandas as pd

class Recommender():
    '''
    Predicts movie ratings using FunkSVD and makes movie recommendations based on predicted ratings
    Provides convenience method for content-based recommendations

    ATTRIBUTES:
        user_mat - (Pandas DataFrame) left-singular matrix from FunkSVD
        movie_mat - (Pandas DataFrame) right-singular matrix from FunkSVD
    '''


    def __init__(self):
        '''
        what do we need to start out our recommender system
        '''
        self.user_mat = None
        self.movie_mat = None


    def fit(self, X, user_ids, movie_ids, latent_features=4, learning_rate=0.0001, iters=100):
        '''
        This function performs matrix factorization using a basic form of FunkSVD with no regularization

        INPUT:
        X - (numpy matrix) a matrix with users as rows, movies as columns, and ratings as values
        user_ids - (array-like) user id values that correspond to the rows of X
        movie_ids - (array-like) movie id values that correspond to the columns of X
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations

        OUTPUT:
        user_mat - (numpy array) a user by latent feature matrix
        movie_mat - (numpy array) a latent feature by movie matrix
        '''

        assert X.shape[0] == len(user_ids), f'Shape of X {X.shape} does not correspond to length of user_id array'
        assert X.shape[1] == len(movie_ids), f'{X.shape} does not correspond to length of movie_id array'

        # Set up useful values to be used through the rest of the function
        n_users = X.shape[0]
        n_movies = X.shape[1]
        num_ratings = n_users * n_movies - np.isnan(X).sum().sum()  # total number of ratings in the matrix
        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(n_users, latent_features)
        movie_mat = np.random.rand(latent_features, n_movies)
        # initialize sse at 0
        sse_accum = 0
        # header for running results
        print("Optimization Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for i in range(iters):
            # update our sse
            old_sse = sse_accum
            sse_accum = 0
            # For each user-movie pair
            for user in range(n_users):
                for movie in range(n_movies):
                    # if the rating exists
                    if not np.isnan(X[user, movie]):
                        # compute the error as the actual minus the dot product of the user and movie latent features
                        error = X[user, movie] - np.dot(user_mat[user, :], movie_mat[:, movie])
                        # Keep track of the total sum of squared errors for the matrix
                        sse_accum += error**2
                        # update the values in each matrix in the direction of the gradient
                        user_mat[user, :] += learning_rate * 2 * error * movie_mat[:, movie]
                        movie_mat[:, movie] += learning_rate * 2 * error * user_mat[user, :]
                        # print results for iteration
            print(f'Iteration {i + 1} | Old MSE: {old_sse / num_ratings:0.2f} \t New MSE:{sse_accum / num_ratings:0.2f}')
        print(f'Final MSE: {sse_accum / num_ratings:0.2f}')

        self.user_mat = pd.DataFrame(user_mat, index=user_ids)
        self.movie_mat = pd.DataFrame(movie_mat, columns=movie_ids)
        return user_mat, movie_mat


    def predict(self, user_ids, movie_ids):
        '''
        makes predictions of a rating for a user on a movie-user combo

        INPUT:
        user_ids - (list) list of user ids for which to predict ratings
        movie_ids - (list) list of movie ids for which to predict ratings
        RETURNS:
        ratings - (numpy ndarray) user-item matrix with elements as ratings
        ratings_index - (numpy array) array of user_ids corresponding to row indices of ratings matrix
        ratings_columns - (numpy array) array of movie_ids corresponding to columns indices of ratings matrix
        '''

        # initialize user-item matrix
        ratings_mat = np.empty((len(user_ids), len(movie_ids)), dtype=float)
        ratings_mat[:] = np.nan
        ratings = pd.DataFrame(ratings_mat, index=user_ids, columns=movie_ids)

        for user in user_ids:
            for movie in movie_ids:
                user_row = self.user_mat.loc[user, :].values
                movie_col = self.movie_mat.loc[:, movie].values
                # Take dot product of that row and column in U and V to make prediction
                try:
                    ratings[user, movie] = np.dot(user_row, movie_col)
                except:
                    continue
        return ratings.values, ratings.index.values, ratings.columns.values


    def make_user_recs(self, ids, rec_num=5):
        '''
        make recommendations given a user id

        INPUT:
        _id - either a user or movie id (int)
        rec_num - number of recommendations to return (int)
        movies - (DataFrame) movies DataFrame that matches columns of train_data.csv
        reviews - (DataFrame) reviews DataFrame that matches columns of movies_clean.csv

        OUTPUT:
        rec_ids - (array) a list or numpy array of recommended movies by id
        rec_names - (array) a list or numpy array of recommended movies by name
        '''

        recs = {}
        ranked = self.predict_popular_movies()
        for _id in ids:
            if self.user_mat.loc[_id, :].isna().sum() < self.user_mat.shape[1]:
                u_row = self.user_mat[_id, :].values
                preds = u_row @ self.movie_mat
                top_n_idx = preds.argsort()[-rec_num:]
                rec_ids = self.movie_mat.columns[top_n_idx]
            else:
                rec_ids = ranked[0:rec_num].index
            recs[_id] = rec_ids
        return recs


    def predict_popular_movies(self):
        '''
        predict most popular movies

        :return: (Pandas Series) predicted movie ratings in descending order with movie ids as index
        '''
        ratings = pd.Series(np.zeros(self.movie_mat.shape[1]), index=self.movie_mat.columns)
        for movie in self.movie_mat.columns:
            try:
                movie_col = self.movie_mat[:, movie]
                pred_rating = (self.user_mat @ movie_col).mean()
                ratings[movie] = pred_rating
            except:
                continue
        ratings.sort(key=lambda x: x[1], reverse=True)
        return ratings


    def make_content_recs(self, movie_id, movie_content=None):
        '''
        INPUT
        movie_id - a movie_id
        movie_content - (csv) a csv containing movie information as columns and movie ids as row index
        OUTPUT
        similar_movies - an array of the most similar movies by title
        '''

        if movie_content is None:
            movie_content = pd.read_csv('movies_clean')
            movie_content = movie_content.set_index('movie_id')
            movie_content = movie_content.iloc[:, 3:]
        else:
            movie_content = pd.read_csv(movie_content)
        # Take the dot product to obtain a movie x movie matrix of similarities
        dot_prod_movies = movie_content @ movie_content.T
        # find movies similar to movie id
        movie_match = dot_prod_movies.loc[movie_id]
        movie_ids = [movie_match.index[i] for i in range(movie_match.shape[0]) if movie_match[i] == movie_match.shape[0]]
        return movie_ids
