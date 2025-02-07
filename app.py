pip install -U scikit-learn

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv") 

    movies = movies.head(288983)

    movies["genres"] = movies["genres"].str.replace("|", " ")

    return movies

def compute_similarity(movies):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  # จำกัดคำสูงสุดที่ 5000 คำ
    genre_vectors = vectorizer.fit_transform(movies["genres"])

    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(genre_vectors)

    return nn, genre_vectors


movies = load_data()

nn_model, genre_vectors = compute_similarity(movies)


def recommend_movie(movie_name, num_recommendations=5):
    movie_idx = movies[movies["title"].str.contains(movie_name, case=False, na=False)].index
    if len(movie_idx) == 0:
        return ["❌ ไม่พบหนังที่ต้องการ กรุณาลองใหม่"]
    movie_idx = movie_idx[0]

    distances, indices = nn_model.kneighbors(genre_vectors[movie_idx])

    recommended_movies = [movies.iloc[i]["title"] for i in indices[0][1:num_recommendations + 1]]
    return recommended_movies


# ==============================
# ส่วนของ Web Application
# ==============================
st.title("🎬 Movie Recommendation System")
st.write("🔎 ป้อนชื่อหนังที่คุณชอบ แล้วระบบจะแนะนำหนังที่คล้ายกันให้!")

movie_name = st.text_input("📌 ป้อนชื่อหนังที่คุณชอบ:", "")

if st.button("แนะนำหนัง 🎥"):
    recommended_movies = recommend_movie(movie_name)
    st.subheader("📌 หนังที่แนะนำ:")
    for idx, movie in enumerate(recommended_movies):
        st.write(f"{idx + 1}. {movie}")
