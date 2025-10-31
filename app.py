import streamlit as st
import pickle
import pandas as pd
import requests
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .movie-card {
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        border: none;
    }
    .recommendation-header {
        font-size: 2rem;
        color: #333;
        margin: 2rem 0 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .similarity-score {
        font-size: 1rem;
        color: #FFD700;
        font-weight: bold;
        background: rgba(0,0,0,0.3);
        padding: 5px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load the precomputed data from pickle files"""
    try:
        movies_df = pickle.load(open('movie_list.pkl', 'rb'))
        similarity_matrix = pickle.load(open('similarity.pkl', 'rb'))
        return movies_df, similarity_matrix
    except FileNotFoundError:
        st.error("Pickle files not found. Please make sure 'movie_list.pkl' and 'similarity.pkl' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def recommend_movies(movie_title, movies_df, similarity_matrix, top_n=5):
    """Get movie recommendations"""
    try:
        # Find the movie index
        movie_indices = movies_df[movies_df['title'].str.lower() == movie_title.lower()].index
        
        if len(movie_indices) == 0:
            # Try partial match
            movie_indices = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower())].index
        
        if len(movie_indices) == 0:
            return None, "Movie not found in database"
        
        index = movie_indices[0]
        selected_movie_title = movies_df.iloc[index].title
        
        # Get similarity scores
        distances = sorted(list(enumerate(similarity_matrix[index])), 
                         reverse=True, key=lambda x: x[1])
        
        recommendations = []
        for i in distances[1:top_n+1]:
            movie_data = {
                'title': movies_df.iloc[i[0]].title,
                'similarity': i[1],
                'index': i[0]
            }
            recommendations.append(movie_data)
        
        return recommendations, selected_movie_title, None
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading movie database...'):
        movies_df, similarity_matrix = load_data()
    
    if movies_df is None or similarity_matrix is None:
        st.error("Failed to load movie data. Please check if the pickle files are available.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 Search Options")
        
        # Movie selection
        movie_titles = sorted(movies_df['title'].tolist())
        
        # Find Avatar or use first movie as default
        default_index = 0
        if 'Avatar' in movie_titles:
            default_index = movie_titles.index('Avatar')
        
        selected_movie = st.selectbox(
            "Choose a movie:",
            movie_titles,
            index=default_index
        )
        
        # Number of recommendations
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=3,
            max_value=10,
            value=5
        )
        
        st.markdown("---")
        st.subheader("Or search by name:")
        search_query = st.text_input("Enter movie name:")
        
        if search_query:
            # Filter movies based on search
            filtered_movies = [title for title in movie_titles if search_query.lower() in title.lower()]
            if filtered_movies:
                selected_movie = st.selectbox("Select from search results:", filtered_movies)
            else:
                st.warning("No movies found with that name.")
        
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.info("""
        This system recommends movies based on:
        - 🎭 Genres
        - 📝 Plot
        - 🎬 Cast
        - 👨‍💼 Directors
        - 🔑 Keywords
        """)

    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Selected Movie")
        
        # Create a card for selected movie
        st.markdown(f"""
        <div class="movie-card">
            <h3 style="margin:0; padding:0;">{selected_movie}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get movie details
        movie_data = movies_df[movies_df['title'] == selected_movie]
        if not movie_data.empty:
            with st.expander("Movie Details"):
                tags_text = movie_data['tags'].iloc[0]
                st.write(f"**Tags:** {tags_text[:300]}..." if len(tags_text) > 300 else f"**Tags:** {tags_text}")
        
        # Recommendation button
        if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
            st.session_state.recommendations_requested = True
            st.session_state.selected_movie = selected_movie
    
    with col2:
        st.subheader("How It Works")
        st.write("""
        ### Content-Based Filtering
        
        This recommendation system uses **content-based filtering** to suggest movies 
        similar to your selection. Here's how it works:
        
        1. **Feature Extraction**: Each movie is converted into a vector based on:
           - Movie genres and categories
           - Plot overview and storyline
           - Cast members (top 3)
           - Director information
           - Keywords and themes
        
        2. **Similarity Calculation**: We use **cosine similarity** to measure how 
           similar movies are to each other based on their feature vectors.
        
        3. **Recommendation**: The system finds movies with the highest similarity 
           scores to your selected movie.
        
        **Why this works**: Movies with similar characteristics, themes, and 
        creative elements tend to appeal to similar audiences!
        """)
    
    # Display recommendations
    if st.session_state.get('recommendations_requested', False) and st.session_state.get('selected_movie') == selected_movie:
        with st.spinner(f'Finding movies similar to "{selected_movie}"...'):
            recommendations, actual_movie_title, error = recommend_movies(
                selected_movie, 
                movies_df, 
                similarity_matrix, 
                num_recommendations
            )
        
        if error:
            st.error(f"❌ {error}")
        elif recommendations:
            st.markdown(f'<h2 class="recommendation-header">🎭 Recommended Movies Similar to "{actual_movie_title}"</h2>', 
                       unsafe_allow_html=True)
            
            # Display recommendations in a grid
            cols = st.columns(2)
            
            for idx, movie in enumerate(recommendations):
                col_idx = idx % 2
                with cols[col_idx]:
                    similarity_percent = movie['similarity'] * 100
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <h3 style="margin:0 0 10px 0;">{movie['title']}</h3>
                        <p class="similarity-score">Match: {similarity_percent:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for visual similarity representation
                    st.progress(float(movie['similarity']), text=f"{similarity_percent:.1f}% similar")
                    
                    # Show movie details on expand
                    with st.expander("View Details"):
                        movie_details = movies_df.iloc[movie['index']]
                        st.write(f"**Full Title:** {movie_details['title']}")
                        st.write(f"**Similarity Score:** {movie['similarity']:.3f}")
                        
                        # Display tags preview
                        tags_text = movie_details['tags']
                        preview_length = min(200, len(tags_text))
                        st.write(f"**Content Tags:** {tags_text[:preview_length]}...")
    
    # Additional features
    with st.expander("📊 System Statistics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movies", f"{len(movies_df):,}")
        
        with col2:
            st.metric("Features Used", "5,000")
        
        with col3:
            st.metric("Similarity Matrix", f"{similarity_matrix.shape[0]:,}²")
        
        with col4:
            avg_similarity = similarity_matrix.mean()
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
    
    with st.expander("🔍 Explore Movie Database", expanded=False):
        st.write("Browse through the movie database:")
        
        # Search and filter
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_term = st.text_input("Search movies:", placeholder="Enter movie name...")
        with search_col2:
            show_count = st.selectbox("Show", [10, 25, 50], index=0)
        
        # Display filtered results
        if search_term:
            filtered_movies = movies_df[movies_df['title'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_movies = movies_df.head(show_count)
        
        st.dataframe(
            filtered_movies[['title']], 
            use_container_width=True,
            height=300
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit | Movie data from TMDB 5000 Dataset</p>
            <p>Recommendation Engine: Content-Based Filtering with Cosine Similarity</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Initialize session state
if 'recommendations_requested' not in st.session_state:
    st.session_state.recommendations_requested = False
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = ""

if __name__ == "__main__":
    main()