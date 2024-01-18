from datetime import datetime
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data_prep.get_last_songs_plus_avg import get_n_last_songs_of_user_by_timestamp, avg_song, get_n_next_songs_of_user_by_timestamp, get_songs_params_by_ids, prepare_songs_df


def load_data(path: str):
    return pd.read_json(path, lines=True)


def prep_train_data():
    songs = load_data('../../data/tracks.jsonl')
    songs_attrs = [
        "popularity",
        "duration_ms",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]
    songs_corr = songs[songs_attrs]
    scaler = StandardScaler().fit(songs_corr)
    songs_normalized = scaler.transform(songs_corr)
    return scaler, songs_normalized, songs


def train_model(training_set: pd.DataFrame, n_neighbors: int = 10):
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(training_set)
    return knn


def prep_test_data(timestamp: pd.Timestamp, user_id: int, n: int = 10):
    sessions_df = load_data("../../data/sessions_clean.jsonl")
    # n songs before the timestamp
    pre_songs = get_n_last_songs_of_user_by_timestamp(sessions_df, user_id, n, timestamp)
    # get details about the picked songs
    songs_corr = prepare_songs_df("../../data/tracks.jsonl")
    pre_songs_params = get_songs_params_by_ids(songs_corr, pre_songs['track_id'].tolist())
    # the thing to base prediction on
    avg = avg_song(pre_songs_params)

    # the things to predict, n songs after the timestamp
    post_songs = get_n_next_songs_of_user_by_timestamp(sessions_df, user_id, n, timestamp)
    return avg, post_songs


def make_prediction(knn: NearestNeighbors, input: pd.DataFrame, scaler: StandardScaler):
    input_norm = scaler.transform(input)
    distances, indices = knn.kneighbors(input_norm)
    return indices


def test_accuracy(prediction, actual_values):
    # get number of songs that are in both lists
    set1 = set(prediction)
    set2 = set(actual_values)
    common_songs = set1.intersection(set2)
    n = len(common_songs)
    s = len(prediction)
    return n/s

def main():
    N = 10
    USER_ID = 111
    DATE = pd.Timestamp(datetime(2023, 8, 4, 2, 38, 38))

    scaler, train_set, songs_df = prep_train_data()
    knn = train_model(train_set)
    input, output = prep_test_data(DATE, USER_ID, N)
    prediction = make_prediction(knn, input, scaler)[0]
    recommended_songs_ids = songs_df.iloc[prediction]['id']
    actual_songs_ids = output['track_id']

    recommended_songs_names = get_songs_params_by_ids(songs_df, recommended_songs_ids.tolist())['name'].tolist()
    actual_songs_names = get_songs_params_by_ids(songs_df, actual_songs_ids.tolist())['name'].tolist()

    test_accuracy(recommended_songs_ids, actual_songs_ids)
    print("Accuracy: ", test_accuracy(recommended_songs_ids, actual_songs_ids))

    print("Prediction: \n", recommended_songs_names)
    print("Actual values: \n", actual_songs_names)


if __name__ == '__main__':
    main()
