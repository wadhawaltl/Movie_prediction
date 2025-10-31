import pickle
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(connect=3, backoff_factor=1, status_forcelist=[502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

API_KEY = "your api"
@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US&append_to_response=videos"
        response = session.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        poster_path = data.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Image"

        overview = data.get("overview", "No description available.")
        release_date = data.get("release_date", "Unknown")
        rating = data.get("vote_average", "N/A")

        trailer_url = None
        videos = data.get("videos", {}).get("results", [])
        for video in videos:
            if video["type"] == "Trailer" and video["site"] == "YouTube":
                trailer_url = f"https://www.youtube.com/embed/{video['key']}"
                break

        return poster_url, overview, release_date, rating, trailer_url

    except requests.exceptions.RequestException as e:
        return "https://via.placeholder.com/500x750?text=No+Image", "N/A", "N/A", "N/A", None

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        title = movies.iloc[i[0]].title
        poster, overview, release_date, rating, trailer = fetch_movie_details(movie_id)
        recommendations.append((title, poster, overview, release_date, rating, trailer))
    return recommendations

st.header('Movie Recommender System')
movies = pickle.load(open('movie_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button('Show Recommendation'):
    recommendations = recommend(selected_movie)
    for title, poster, overview, release_date, rating, trailer in recommendations:
        st.markdown("---")
        
        # Poster and trailer side by side
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(poster, use_container_width=True)
        with col2:
            if trailer:
                st.video(trailer)

        # Movie details below
        st.markdown(f"### {title}")
        st.caption(f"Release Date:{release_date} | Rating:{rating}")
        st.write(overview if len(overview) < 400 else overview[:400] + "...")
