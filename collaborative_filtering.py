# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# loading csv file into pandas dataframe

# specify path
my_path = "ratings.csv"

# read ratings file
ratings = pd.read_csv(my_path + 'ratings.csv')

ratings.head()
ratings.tail()
ratings.shape
ratings.info()


X_train, X_test = train_test_split(ratings, test_size = 0.30, random_state = 42)

print(X_train.shape)
print(X_test.shape)

# pivot ratings into movie features
user_data = X_train.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
user_data.head()

# make a copy of train and test datasets
dummy_train = X_train.copy()
dummy_test = X_test.copy()

dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)

# The movies not rated by user is marked as 1 for prediction 
dummy_train = dummy_train.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(1)

# The movies not rated by user is marked as 0 for evaluation 
dummy_test = dummy_test.pivot(index ='userId', columns = 'movieId', values = 'rating').fillna(0)

dummy_train.head()
dummy_test.head()

# users - user similarity matrix (Using Cosine Similarity)
# User Similarity Matrix using Cosine similarity as a similarity measure between Users
user_similarity = cosine_similarity(user_data)
user_similarity[np.isnan(user_similarity)] = 0
print(user_similarity)
print(user_similarity.shape)

# Predicting user ratings on the movies
user_predicted_ratings = np.dot(user_similarity, user_data)
print(user_predicted_ratings)

# np.multiply for cell-by-cell multiplication 

user_final_ratings = np.multiply(user_predicted_ratings, dummy_train)
user_final_ratings.head()

# Top 5 movie recommendations for user 42
user_final_ratings.iloc[42].sort_values(ascending = False)[0:5]

# Item Based Collaborative Filtering
movie_features = X_train.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
movie_features.head()

# Item - Item Similarity Matrix
# Item Similarity Matrix using Cosine similarity as a similarity measure between Items
item_similarity = cosine_similarity(movie_features)
item_similarity[np.isnan(item_similarity)] = 0
print(item_similarity)
print("- "*10)
print(item_similarity.shape)

# Predicting user ratings for the movies
item_predicted_ratings = np.dot(movie_features.T, item_similarity)
item_predicted_ratings

# Filtering the ratings only for the movies not rated by the user for recommendation
# np.multiply for cell-by-cell multiplication 

item_final_ratings = np.multiply(item_predicted_ratings, dummy_train)
item_final_ratings.head()

# Top 5 movie recommendations for user 42
item_final_ratings.iloc[42].sort_values(ascending = False)[0:5]

# Evaluation
# Using User-User similarity
test_user_features = X_test.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
test_user_similarity = cosine_similarity(test_user_features)
test_user_similarity[np.isnan(test_user_similarity)] = 0

print(test_user_similarity)
print("- "*10)
print(test_user_similarity.shape)
user_predicted_ratings_test = np.dot(test_user_similarity, test_user_features)
print(user_predicted_ratings_test)

# Testing on the movies already rated by the user
test_user_final_rating = np.multiply(user_predicted_ratings_test, dummy_test)
test_user_final_rating.head()
ratings['rating'].describe()

# MinMax Scaling
X = test_user_final_rating.copy() 
X = X[X > 0] # only consider non-zero values as 0 means the user haven't rated the movies

scaler = MinMaxScaler(feature_range = (0.5, 5))
scaler.fit(X)
pred = scaler.transform(X)

print(pred)

# total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(pred))
print(total_non_nan)
test = X_test.pivot(index = 'userId', columns = 'movieId', values = 'rating')
test.head()

# RMSE Score

diff_sqr_matrix = (test - pred)**2
sum_of_squares_err = diff_sqr_matrix.sum().sum() # df.sum().sum() by default ignores null values

rmse = np.sqrt(sum_of_squares_err/total_non_nan)
print(rmse)

# Mean abslute error

mae = np.abs(pred - test).sum().sum()/total_non_nan
print(mae)

# Using Item-Item similarity
test_item_features = X_test.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
test_item_similarity = cosine_similarity(test_item_features)
test_item_similarity[np.isnan(test_item_similarity)] = 0 

print(test_item_similarity)
print("- "*10)
print(test_item_similarity.shape)

item_predicted_ratings_test = np.dot(test_item_features.T, test_item_similarity )
print(item_predicted_ratings_test)

# Testing on the movies already rated by the user
test_item_final_rating = np.multiply(item_predicted_ratings_test, dummy_test)
test_item_final_rating.head()
ratings['rating'].describe()

# MinMax Scaling
X = test_item_final_rating.copy() 
X = X[X > 0] # only consider non-zero values as 0 means the user haven't rated the movies

scaler = MinMaxScaler(feature_range = (0.5, 5))
scaler.fit(X)
pred = scaler.transform(X)

print(pred)

# total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(pred))
print(total_non_nan)
test = X_test.pivot(index = 'userId', columns = 'movieId', values = 'rating')
test.head()

# RMSE Score

diff_sqr_matrix = (test - pred)**2
sum_of_squares_err = diff_sqr_matrix.sum().sum() # df.sum().sum() by default ignores null values

rmse = np.sqrt(sum_of_squares_err/total_non_nan)
print(rmse)

# Mean abslute error

mae = np.abs(pred - test).sum().sum()/total_non_nan
print(mae)