import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Prepare user-movie matrix
final_dataset = ratings.pivot(index="movieId", columns="userId", values="rating")
final_dataset.fillna(0, inplace=True)

# Filter movies rated by more than 10 users
no_user_voted = ratings.groupby("movieId")["rating"].count()
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]

# Filter users who have rated more than 50 movies
no_movies_voted = ratings.groupby("userId")["rating"].count()
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Convert to sparse matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Fit k-NN model
knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Recommendation function
def get_recommendation(movie_name):
    movie_list = movies[movies["title"].str.contains(movie_name, case=False, na=False)]
    
    if not movie_list.empty:
        movie_idx = movie_list.iloc[0]["movieId"]
        
        if movie_idx not in final_dataset["movieId"].values:
            return "Movie found, but not enough data for recommendation."

        movie_idx = final_dataset[final_dataset["movieId"] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=11)

        recommended_movies = []
        for i in range(1, len(indices.flatten())):
            idx = indices.flatten()[i]
            recommended_movie_id = final_dataset.iloc[idx]["movieId"]
            movie_title = movies[movies["movieId"] == recommended_movie_id]["title"].values[0]
            recommended_movies.append({"Title": movie_title, "Similarity Score": 1 - distances.flatten()[i]})
        
        return pd.DataFrame(recommended_movies, index=range(1, 11))
    else:
        return "Movie not found."

# Streamlit App UI
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.markdown("Type a movie name to get similar movie recommendations!")

movie_input = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    if movie_input.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        result = get_recommendation(movie_input)
        if isinstance(result, str):
            st.error(result)
        else:
            st.success(f"Movies similar to **{movie_input}**:")
            st.dataframe(result)
