from flask import Flask, request, jsonify
from surprise import Dataset, Reader, SVD
import pandas as pd
from target_model.knn_model import KNNModel
from base_model.model import MostListenedTracksModel, MostPopularTracksModel

app = Flask(__name__)

import pickle

model_knn = KNNModel()
model_popular = MostPopularTracksModel()
model_most_listened = MostListenedTracksModel()

try:
    with open('knn_model.pkl', 'rb') as file:
        model_knn = model_knn.load_model(file)
except:
    pass

try:
    with open('popular_model.pkl', 'rb') as file:
        model_popular = model_popular.load(file)
except:
    pass   

try:
    with open('listened_model.pkl', 'rb') as file:
        model_most_listened = model_most_listened.load(file)
except:
    pass


def make_a_ids_list(df):
    ids = []
    for i in df.index:
        ids.append(df.loc[i]['id'])
    return ids

@app.route('/predict/knn', methods=['POST'])
def predict_knn():
    try:
        # Get user ID and songs from the request
        songs = request.json['songs']
        songs = pd.DataFrame(songs)
        merged_song = model_knn.avg_song(songs)


        # Make predictions
        predictions = model_knn.predict(merged_song)

        # Return the top N recommendations

        return jsonify({'recommendations': make_a_ids_list(predictions)})

    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/predict/popular', methods=['POST'])
def predict_popular():
    try:
    
        predictions = model_popular.predict(10)

        return jsonify({'recommendations': make_a_ids_list(predictions)})

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/predict/most_listened', methods=['POST'])
def predict_most_listened():
    try:
        # Get user ID and songs from the request
        genres = request.json['genres']
        # Make predictions
        predictions = model_most_listened.predict(genres)
        # Return the top N recommendations
        return jsonify({'recommendations': make_a_ids_list(predictions)})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/train/knn', methods=['POST'])
def train_knn():
    try:
        songs = request.json['songs']
        songs = pd.DataFrame(songs)
        # Make predictions
        preprocessed_songs = model_knn.fit_data_preprocessor(songs)
        model_knn.fit(preprocessed_songs)
        return

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/train/popular', methods=['POST'])
def train_popular():
    try:
        songs = request.json['songs']
        songs = pd.DataFrame(songs)
        # Make predictions
        model_popular.train(songs, 10)
        return

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/train/most_listened', methods=['POST'])
def train_most_listened():
    try:
        users = pd.DataFrame(request.json['users'])
        sessions = pd.DataFrame(request.json['sessions'])
        model_most_listened.train(users, sessions, 10)
        

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/save/knn', methods=['POST'])
def save_knn():
    model_knn.save_model('models/knn_model.pkl')

@app.route('/save/popular', methods=['POST'])
def save_popular():
    model_popular.save('models/popular_model.pkl')

@app.route('/save/most_listened', methods=['POST'])
def save_most_listened():
    model_most_listened.save('models/listened_model.pkl')

@app.route('/predict/abc', methods=['POST'])
def predict_abc():
    try:
        # Get user ID and songs from the request
        genres = request.json['genres']
        songs = request.json['songs']
        songs = pd.DataFrame(songs)
        merged_song = model_knn.avg_song(songs)
        knn_predictions = model_knn.predict(merged_song)
        results = {
            'knn_predictions': make_a_ids_list(knn_predictions),
            'popular_predictions': make_a_ids_list(model_popular.predict(10)),
            'listened_predictions': make_a_ids_list(model_most_listened.predict(genres))
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})




if __name__ == '__main__':
    app.run(debug=True)
