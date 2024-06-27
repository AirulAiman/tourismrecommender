from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('content_based_model.pkl', 'rb') as f:
    cosine_sim_df, tourism_new = pickle.load(f)

def tourism_recommendations(place_name, similarity_data=cosine_sim_df, items=tourism_new, k=5):
    if place_name not in similarity_data.columns:
        return []
    similarities = similarity_data.loc[:, place_name]
    if len(similarities) < k + 1:
        k = len(similarities) - 1

    index = similarities.to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(place_name, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k).to_dict(orient='records')

@app.route('/recommend', methods=['GET'])
def recommend():
    place_name = request.args.get('place_name')
    if not place_name:
        return jsonify({"error": "Missing 'place_name' parameter"}), 400
    recommendations = tourism_recommendations(place_name)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
