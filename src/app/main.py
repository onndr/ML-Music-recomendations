from flask import Flask, request, jsonify
import pandas as pd
from src.target_model.knn_model import KNNModel
from src.base_model.model import MostListenedTracksModel, MostPopularTracksModel

app = Flask(__name__)

model_knn = KNNModel()
model_popular = MostPopularTracksModel()
model_most_listened = MostListenedTracksModel()

try:
    model_knn.load_model('models/knn_model.pkl')
except:
    pass

try:
    model_popular.load('models/most_popular_model.jsonl')
except:
    pass

try:
    model_most_listened.load('models/most_listened_model.jsonl')
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
        songs = request.json['songs']
        songs_df = pd.DataFrame(songs)
        songs_preprocessed = model_knn.preprocess_data(songs_df)
        merged_song = model_knn.avg_song(songs_preprocessed)
        predictions = model_knn.predict(merged_song)

        return jsonify({'recommendations': make_a_ids_list(predictions)})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict/popular', methods=['POST'])
def predict_popular():
    try:
        predictions = model_popular.predict(10)

        return jsonify({'recommendations': predictions})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict/most_listened', methods=['POST'])
def predict_most_listened():
    try:
        genres = request.json['genres']
        predictions = model_most_listened.predict(genres)
        return jsonify({'recommendations': predictions})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/train/knn', methods=['POST'])
def train_knn():
    try:
        songs = request.json['songs']
        songs = pd.DataFrame(songs)
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
    model_popular.save('models/most_popular_model.jsonl')


@app.route('/save/most_listened', methods=['POST'])
def save_most_listened():
    model_most_listened.save('models/listened_model.jsonl')


@app.route('/predict/abc', methods=['POST'])
def predict_abc():
    try:
        genres = request.json['genres']
        songs = request.json['songs']
        songs_df = pd.DataFrame(songs)
        songs_preprocessed = model_knn.preprocess_data(songs_df)
        merged_song = model_knn.avg_song(songs_preprocessed)
        knn_predictions = model_knn.predict(merged_song)
        results = {
            'knn_recommendations': make_a_ids_list(knn_predictions),
            'popular_recommendations': model_popular.predict(10),
            'listened_recommendations': model_most_listened.predict(genres)
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
