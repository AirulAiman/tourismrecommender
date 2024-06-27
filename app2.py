import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import joblib
from IPython.display import display, HTML
from model import RecommenderNet

# Load the saved data and model
data = joblib.load('tourism_data.joblib')
cosine_sim_df = data['cosine_sim_df']
all_tourism = data['all_tourism']
tourism_rating = data['tourism_rating']
place_to_place_encoded = data['place_to_place_encoded']
user_to_user_encoded = data['user_to_user_encoded']
place_encoded_to_place = data['place_encoded_to_place']
model = tf.keras.models.load_model('tourism_model.keras', custom_objects={'RecommenderNet': RecommenderNet})

def tourism_Recommendations(place_name, similarity_data=cosine_sim_df, items=all_tourism[['Title', 'Genre', 'GoogleMapsLink']], k=5):
    if place_name not in similarity_data.columns:
        raise ValueError(f"Place name '{place_name}' not found in similarity data.")

    similarities = similarity_data.loc[:, place_name]
    if len(similarities) < k + 1:
        k = len(similarities) - 1

    if k < 1:
        raise ValueError(f"Not enough similar places found for '{place_name}' to make recommendations.")

    index = similarities.to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(place_name, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

def hybrid_recommendations(place_name, user_id, content_weight=0.5, collab_weight=0.5, top_k=10):
    padded_place_name = place_name.ljust(5)
    place_titles = all_tourism['Title'].tolist()
    best_match, score = process.extractOne(padded_place_name, place_titles)

    try:
        content_recs = tourism_Recommendations(best_match)
        content_scores = {row['Title']: cosine_sim_df[best_match][row['Title']] for index, row in content_recs.iterrows()}
    except ValueError as e:
        st.error(str(e))
        return pd.DataFrame()

    place_df = all_tourism
    df = tourism_rating.copy()
    place_visited_by_user = df[df['Rating'] == user_id]
    place_not_visited = place_df[~place_df['ID'].isin(place_visited_by_user['ID'].values)]['ID']
    place_not_visited = list(set(place_not_visited).intersection(set(place_to_place_encoded.keys())))
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_encoder = user_to_user_encoded.get(user_id)

    if not place_not_visited or user_encoder is None:
        st.warning("No recommendations available for this user and place combination.")
        return pd.DataFrame()

    # Ensure user_place_array is properly handled for None values
    user_place_array = np.array(place_not_visited)
    user_place_array = np.nan_to_num(user_place_array, nan=0)
    user_place_array = np.hstack(([[user_encoder]] * len(user_place_array), user_place_array))
    
    ratings = model.predict(user_place_array).flatten()
    collab_scores = {place_encoded_to_place.get(place_not_visited[x][0]): ratings[x] for x in range(len(place_not_visited))}

    combined_scores = {}
    for place, score in content_scores.items():
        combined_scores[place] = content_weight * score + collab_weight * collab_scores.get(place_df[place_df['Title'] == place]['ID'].values[0], 0)
    for place, score in collab_scores.items():
        title = place_df[place_df['ID'] == place]['Title'].values[0]
        if title not in combined_scores:
            combined_scores[title] = collab_weight * score

    sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = sorted_scores[:top_k]

    recommended_titles = []
    seen_titles = set()
    for title, _ in top_recommendations:
        if title not in seen_titles:
            recommended_titles.append(title)
            seen_titles.add(title)

    recommended_titles = [title for title in recommended_titles if title in place_df['Title'].values]
    recommendations = place_df[place_df['Title'].isin(recommended_titles)][['Title', 'Genre', 'GoogleMapsLink']]
    recommendations['GoogleMapsLink'] = recommendations['GoogleMapsLink'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

    st.markdown("### Recommended Places")
    st.write(recommendations)
    return recommendations


def main():
    st.title("Tourism Recommendation System")
    
    # Get the list of titles
    tour = all_tourism['Title'].values
    
    # Create a selectbox for user input
    selected_tour = st.selectbox('Select Place', tour)
    
    user_id = st.selectbox("Select User ID", tourism_rating['Rating'].unique())
    st.markdown("---")
    
    if st.button("Get Recommendations"):
        hybrid_recommendations(selected_tour, user_id)

if __name__ == '__main__':
    main()
