import pickle
import streamlit as st
import pandas as pd  # Import pandas
import requests
import sys
from pathlib import Path

st.header("Tourism Recommendation System Using Machine Learning")



all_tourism = Path(__file__).parents[1] / 'model/contentbased.pkl'

cosine_sim_df = Path(__file__).parents[1] / 'model/contentbased.pkl'

# Get the list of titles
tour = all_tourism['Title'].values

# Create a selectbox for user input
selected_tour = st.selectbox('Type Location', tour)

# Function to recommend similar places
def tourism_recommendations(place_name, similarity_data=cosine_sim_df, items=all_tourism[['Title', 'Genre', 'Rating', 'GoogleMapsLink']], k=5):
    if place_name not in similarity_data.columns:
        st.write(f"Place name '{place_name}' not found in similarity data.")
        return pd.DataFrame()
    
    similarities = similarity_data.loc[:, place_name]
    if len(similarities) < k + 1:
        k = len(similarities) - 1

    index = similarities.to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(place_name, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

# Display the recommendations if a tour is selected
if selected_tour:
    recommendations = tourism_recommendations(selected_tour)
    for index, row in recommendations.iterrows():
        st.markdown(f"### {row['Title']}")
        st.write(f"**Genre:** {row['Genre']}")
        st.write(f"**Rating:** {row['Rating']}")
        st.write(f"linkLocation: {row['GoogleMapsLink']}")
        # Embed Google Maps using an iframe
        st.markdown(f'<iframe src="{row["GoogleMapsLink"]}" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe>', unsafe_allow_html=True)

