import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def featurize_data(idx, user, movie, rating, sparse_matrix, avgs_dict):
    
    sampled_train_averages = avgs_dict
    #------------------ratings of movie by similar users for "current user"-------------------
    top_sim_users = cosine_similarity(sparse_matrix[user], sparse_matrix).ravel().argsort()[::-1]
    top_sim_user_ratings = sparse_matrix[top_sim_users, movie].toarray().ravel()
    top_sim_user_ratings = top_sim_user_ratings[top_sim_user_ratings != 0][:5].tolist()
    top_sim_user_ratings += [sampled_train_averages['movie'][movie]]*(5 - len(top_sim_user_ratings))
    #print(user)
        
    #------------------ratings of user to similar movies of 'cuurent movie'---------------------
    top_sim_movies = cosine_similarity(sparse_matrix[:, movie].T, sparse_matrix.T).ravel().argsort()[::-1]
    top_sim_movie_ratings = sparse_matrix[user, top_sim_movies].toarray().ravel()
    top_sim_movie_ratings = top_sim_movie_ratings[top_sim_movie_ratings != 0][:5].tolist()
    top_sim_movie_ratings += [sampled_train_averages['user'][user]]*(5 - len(top_sim_movie_ratings))
    #print(movie)
        
    #------------------build a list using the above features-----------------------------
        
    vector = list()
    vector.append(user)
    vector.append(movie)
    vector.append(sampled_train_averages['global'])
    vector += top_sim_user_ratings
    vector += top_sim_movie_ratings
    vector.append(sampled_train_averages['user'][user])
    vector.append(sampled_train_averages['movie'][movie])
    vector.append(rating)
        
    return (idx, vector)
    