import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------------- Load Dataset -------------------
movies = pd.read_csv("top10K-TMDB-movies.csv")
movies = movies[['id', 'title', 'overview']].dropna()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ------------------- TMDB API Setup -------------------
API_KEY = "d9a1bb32cb6582181cdae7bfe8a53046"  # Replace with your TMDB API key
session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

def fetch_movie_details(movie_id):
    """Fetch poster, overview, and rating from TMDB API with graceful fallback"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Poster fallback
        poster_path = data.get("poster_path")
        poster = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/250x375.png?text=No+Image"

        # Overview fallback
        overview = data.get("overview")
        if not overview:
            overview = "Overview not available."

        # Rating fallback
        rating = data.get("vote_average")
        if not rating:
            rating = "N/A"

        return poster, overview, rating

    except Exception:
        # Fallback for network/API errors
        poster = "https://via.placeholder.com/250x375.png?text=No+Image"
        overview = "Overview not available."
        rating = "N/A"
        return poster, overview, rating

# ------------------- Recommendation Function -------------------
def get_recommendations(title, cosine_sim=cosine_sim, top_n=5):
    """Get top N similar movies based on cosine similarity"""
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # exclude itself
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# ------------------- Streamlit UI -------------------
st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get similar recommendations with posters, overview, and ratings!")

selected_movie = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie, top_n=5)
    
    if len(recommendations) == 0:
        st.error("Movie not found in dataset!")
    else:
        st.write(f"### Recommendations for **{selected_movie}**:")

        for _, row in recommendations.iterrows():
            poster, overview, rating = fetch_movie_details(row['id'])

            # Movie card layout
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 30px;">
                    <img src="{poster}" width="250"><br>
                    <h3>{row['title']}</h3>
                    <p><b>⭐ Rating:</b> {rating}/10</p>
                    <p style="max-width: 600px; margin: auto;">📝 {overview}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("---")
