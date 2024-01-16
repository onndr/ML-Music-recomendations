from flask import Flask, request, jsonify
from surprise import Dataset, Reader, SVD
import pandas as pd

app = Flask(__name__)

import pickle

# Load the model from the file
with open('svd_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user ID and songs from the request
        user_id = request.json['user_id']
        songs = request.json['songs']

        # Make predictions
        predictions = [(user_id, song, model.predict(user_id, song).est) for song in songs]

        # Return the top N recommendations
        top_n = sorted(predictions, key=lambda x: x[2], reverse=True)[:10]
        recommended_song_ids = [song for _, song, _ in top_n]

        return jsonify({'recommendations': recommended_song_ids})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
