import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print("Movies Loaded:", movies.shape)
print("Ratings Loaded:", ratings.shape)

# ---------------------------------------------------------
# CONTENT-BASED FILTERING
# ---------------------------------------------------------
print("\nBuilding Content-Based Model...")

# Combine genre text
movies['genres'] = movies['genres'].str.replace('|', ' ')

# Vectorize genres
vectorizer = CountVectorizer(stop_words='english')
genre_matrix = vectorizer.fit_transform(movies['genres'])

# Compute cosine similarity
similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)

def recommend_content(movie_name, top_n=5):
    if movie_name not in movies['title'].values:
        return f"Movie '{movie_name}' not found."
    
    # Get movie index
    idx = movies[movies['title'] == movie_name].index[0]
    
    # Get similarity scores
    scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort scores
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommended_movies = [movies.iloc[i[0]].title for i in sorted_scores]
    return recommended_movies

print("\nContent Filtering Ready!")

# ---------------------------------------------------------
# COLLABORATIVE FILTERING (SVD)
# ---------------------------------------------------------
print("\nTraining Collaborative Filtering Model...")

# Define reader
reader = Reader(rating_scale=(0.5, 5.0))

# Load dataset
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Apply SVD
model = SVD()
model.fit(trainset)

print("Collaborative Model Ready!")

def recommend_collaborative(user_id, top_n=5):
    user_ratings = ratings[ratings['userId'] == user_id]
    seen_movies = user_ratings['movieId'].unique()
    
    # Predict rating for all unseen movies
    movie_ids = movies['movieId'].unique()
    predictions = []
    
    for movie_id in movie_ids:
        if movie_id not in seen_movies:
            pred = model.predict(user_id, movie_id).est
            predictions.append((movie_id, pred))
    
    # Sort top predictions
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    recommended_titles = [
        movies[movies['movieId'] == movie_id].title.values[0]
        for movie_id, _ in top_movies
    ]
    
    return recommended_titles

# ---------------------------------------------------------
# TEST THE MODELS
# ---------------------------------------------------------

print("\n==============================")
print(" EXAMPLE RECOMMENDATIONS ")
print("==============================\n")

# Content-based Example
movie_query = "Toy Story (1995)"
print("Content-based recommendations for:", movie_query)
print(recommend_content(movie_query), "\n")

# Collaborative Example
print("Collaborative Filtering recommendations for User 1:")
print(recommend_collaborative(1))
