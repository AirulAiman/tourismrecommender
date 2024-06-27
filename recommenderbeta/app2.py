import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load your data and model
info_tourism = pd.read_csv("TravelMyINFO.csv")
tourism_rating = pd.read_csv("TravelMyRATING.csv")
model = load_model('hybrid_recommendation_model_weights.pkl')  # Replace with the path to your saved TensorFlow model

# Merge the datasets
all_tourism = pd.merge(tourism_rating, info_tourism[['ID', 'Title']], on='ID', how='left')

# Create a function for recommendations
def hybrid_recommendations(place_name):
    # Function to find closest matching place name using fuzzy matching
    def find_closest_place_name(place_name):
        place_titles = all_tourism['Title'].tolist()
        best_match, score = process.extractOne(place_name, place_titles)
        return best_match
    
    # Find the closest matching place name
    place_name = find_closest_place_name(place_name)
    
    # Placeholder for recommendation logic based on your model
    # Replace this with your actual recommendation logic
    # Example: Get similar places based on your hybrid model
    similar_places = ["Similar Place 1", "Similar Place 2", "Similar Place 3"]  # Replace with actual recommendations
    
    return similar_places

# Streamlit UI
def main():
    st.title('Hybrid Recommendation System')
    
    # Input field for place name
    place_name = st.text_input('Enter a place name:')
    
    # Button to trigger recommendations
    if st.button('Get Recommendations'):
        if place_name:
            recommendations = hybrid_recommendations(place_name)
            st.write('Recommended Places:')
            st.write(recommendations)
        else:
            st.write('Please enter a place name.')

if __name__ == '__main__':
    main()
