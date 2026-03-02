"""
RankFlix Recommender - Streamlit App
"""

import streamlit as st
import pandas as pd
from recommender import get_recommender

# Page config
st.set_page_config(
    page_title="RankFlix Recommender",
    page_icon="🎬",
    layout="wide"
)

# Load recommender (cached)
@st.cache_resource
def load_recommender():
    return get_recommender("models")

@st.cache_data
def load_movies():
    return pd.read_csv("archive/movies.csv")

# Initialize
recommender = load_recommender()
movies_df = load_movies()

# Helper function to get movie details
def get_movie_details(movie_ids: list) -> pd.DataFrame:
    details = movies_df[movies_df["movieId"].isin(movie_ids)].copy()
    # Preserve recommendation order
    details["rank"] = details["movieId"].apply(lambda x: movie_ids.index(x) + 1)
    details = details.sort_values("rank")
    return details[["rank", "movieId", "title", "genres"]]

# Title
st.title("🎬 RankFlix Recommender")
st.markdown("A two-stage hybrid recommender using tag-based semantic matching and LightGBM ranking.")

st.divider()

# Sidebar for input method selection
st.sidebar.header("Recommendation Mode")
mode = st.sidebar.radio(
    "Choose input method:",
    ["🔥 Popular (Anonymous)", "🏷️ By Tags", "🎥 By Liked Movies"],
    index=0
)

# Number of recommendations
k = st.sidebar.slider("Number of recommendations", min_value=5, max_value=50, value=10)

st.sidebar.divider()
st.sidebar.markdown("### About")
st.sidebar.markdown("""
**Stage 1**: Candidate generation via popularity and tag similarity

**Stage 2**: LightGBM LambdaRank scoring
""")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input")
    
    input_data = {}
    
    if mode == "🔥 Popular (Anonymous)":
        st.info("No input needed. Showing popular movies.")
        input_data = {}
        
    elif mode == "🏷️ By Tags":
        tags_input = st.text_area(
            "Enter tags (one per line or comma-separated):",
            placeholder="sci-fi\naction\nthriller",
            height=150
        )
        if tags_input:
            # Parse tags (handle both newlines and commas)
            tags = [t.strip() for t in tags_input.replace(",", "\n").split("\n") if t.strip()]
            input_data = {"liked_tags": tags}
            st.write("**Parsed tags:**", tags)
        else:
            st.warning("Enter some tags to get personalized recommendations.")
            
    elif mode == "🎥 By Liked Movies":
        # Show some sample movies for selection
        st.markdown("**Search and select movies you like:**")
        
        search_query = st.text_input("Search movies:", placeholder="Type movie title...")
        
        if search_query:
            matches = movies_df[
                movies_df["title"].str.contains(search_query, case=False, na=False)
            ].head(20)
            
            if len(matches) > 0:
                selected = st.multiselect(
                    "Select movies:",
                    options=matches["movieId"].tolist(),
                    format_func=lambda x: movies_df[movies_df["movieId"] == x]["title"].values[0]
                )
                if selected:
                    input_data = {"liked_movies": selected}
            else:
                st.warning("No movies found.")
        else:
            st.info("Search for movies to select.")

with col2:
    st.subheader("Recommendations")
    
    # Generate recommendations button
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            # Get recommendations
            recommended_ids = recommender.recommend_online(input_data, k=k)
            
            if recommended_ids:
                # Get movie details
                results = get_movie_details(recommended_ids)
                
                # Display results
                st.success(f"Found {len(results)} recommendations!")
                
                # Display as a nice table
                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "rank": st.column_config.NumberColumn("Rank", width="small"),
                        "movieId": st.column_config.NumberColumn("ID", width="small"),
                        "title": st.column_config.TextColumn("Title", width="large"),
                        "genres": st.column_config.TextColumn("Genres", width="medium"),
                    }
                )
            else:
                st.error("No recommendations found.")
    else:
        st.info("Click the button above to generate recommendations.")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit • Two-Stage Hybrid Recommender</div>",
    unsafe_allow_html=True
)
