import pickle
import streamlit as st
import pandas as pd
from fuzzywuzzy import process

# Function to load model data
@st.cache(allow_output_mutation=True)
def load_model_data(file_path):
    with open(file_path, 'rb') as f:
        cosine_sim_df, all_tourism = pickle.load(f)
    return cosine_sim_df, all_tourism

# Function to recommend similar places using fuzzy matching
def tourism_recommendations(place_name, similarity_data, items, k=20):
    closest_match, score = process.extractOne(place_name, similarity_data.columns)
    
    if closest_match not in similarity_data.columns:
        st.warning(f"Place name '{closest_match}' not found in similarity data.")
        return pd.DataFrame()
    
    similarities = similarity_data.loc[:, closest_match]
    k = min(k, len(similarities) - 1)

    index = similarities.argsort()[-k-1:-1][::-1]
    closest = similarity_data.columns[index]
    closest = closest.drop(closest_match, errors='ignore')
    return pd.DataFrame(closest).merge(items)

# Function to convert numerical rating to star representation
def stars_from_rating(rating):
    full_stars = int(rating)
    half_star = (rating - full_stars) >= 0.5
    empty_stars = 5 - full_stars - (1 if half_star else 0)

    stars = '★' * full_stars + '½' * (half_star) + '☆' * empty_stars
    return stars

# Main function to run the Streamlit app
def main():
    st.title("Tourism Recommendation System")
    st.title("Search Options")
    st.text("Input any location full name Keyword that you want to visit (Eg. Hiking , Western , Hotel, Resort etc)")

    # Load model data
    cosine_sim_df, all_tourism = load_model_data('model/contentbased.pkl')

    # User input for location
    selected_tour = st.text_input("Enter Location or type of place that you want to visit", "")
    
    # User input for location
    selected_tour = st.sidebar.text_input("Enter Location or type of place that you want to visit", "")

    if selected_tour:
        recommendations = tourism_recommendations(selected_tour, cosine_sim_df, all_tourism[['Title', 'Genre', 'Rating', 'GoogleMapsLink']])
        st.subheader(f"Top {len(recommendations)} Recommendations for '{selected_tour}'")
        
        for index, row in recommendations.iterrows():
            st.markdown(f"### {row['Title']}")
            st.write(f"**Genre:** {row['Genre']}")
            st.write(f"**Rating:** {row['Rating']} ({stars_from_rating(row['Rating'])})")

            # Display Google Maps link as a button
            st.markdown(f'<a href="{row["GoogleMapsLink"]}" target="_blank" class="button">View Map</a>', unsafe_allow_html=True)

            # Display star rating using FontAwesome
            st.markdown("""
            <style>
            .stars {
                color: gold;
                font-size: large;
                margin-top: 10px;
            }
            .button {
                background-color: #4CAF50;
                border: none;
                color: white;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin-top: 10px;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .button:hover {
                background-color: #45a049;
            }
            </style>
            """, unsafe_allow_html=True)

            # Display star rating
            st.markdown(f'<p class="stars">{stars_from_rating(row["Rating"])}</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
