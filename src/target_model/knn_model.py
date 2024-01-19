import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

songs_attrs_global = [
    "id",
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


class KNNModel:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.attributes_list = None
        self.index = None
        self.all_songs = None
        self.n = None

    def fit(self, songs_normalized, n_neighbors=10):
        print("Start fitting")
        self.n = n_neighbors
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(songs_normalized)
        print("Finished fitting")
        self.model = knn

    def fit_data_preprocessor(self, all_songs, attributes_list=songs_attrs_global, index="id"):
        self.all_songs = all_songs
        self.index = index
        self.attributes_list = attributes_list
        songs_corr = self.all_songs[self.attributes_list]
        songs_corr = songs_corr.set_index(self.index)
        scaler = StandardScaler().fit(songs_corr)
        songs_normalized = scaler.transform(songs_corr)
        self.scaler = scaler
        return songs_normalized

    def preprocess_data(self, songs_df):
        if not self.scaler:
            raise Exception("Data preprocessor not fitted")
        songs_df = songs_df[self.attributes_list]
        songs_df = songs_df.set_index(self.index)
        return self.scaler.transform(songs_df)

    def load_data(self, path: str):
        return pd.read_json(path, lines=True)

    def avg_song(songs: pd.DataFrame):
        return songs.mean(numeric_only=True).to_frame().transpose()

    def predict(self, song: pd.DataFrame):
        if not self.model:
            raise Exception("Model not fitted")
        distances, indices = self.model.kneighbors(song)
        recommended_songs = self.all_songs.iloc[indices[0]]
        return recommended_songs


if __name__ == "__main__":

    knn_model = KNNModel()
    songs = knn_model.load_data('data/tracks.jsonl')
    songs_after_preprocessing = knn_model.fit_data_preprocessor(songs)
    knn_model.fit(songs_after_preprocessing)
    coldplay_yellow = {"id": "4gzpq5DPGxSnKTe4SA8HAU", "popularity": 86, "duration_ms": 266773, "explicit": 0,
                       "danceability": 0.429, "energy": 0.661, "key": 11, "loudness": -7.227, "speechiness": 0.0281,
                       "acousticness": 0.00239, "instrumentalness": 0.000121, "liveness": 0.234, "valence": 0.285,
                       "tempo": 173.372}
    coldplay_yellow = knn_model.preprocess_data(pd.DataFrame([coldplay_yellow]))
    results = knn_model.predict(coldplay_yellow)
    for i in results.index:
        print(results.loc[i]['name'])
    pass

# songs_corr = songs[songs_attrs]
# songs_corr = songs_corr.set_index("id")
# scaler = StandardScaler().fit(songs_corr)
# songs_normalized = scaler.transform(songs_corr)
# print("Start fitting")
# # 3. Create kNN model and fit it to the data
# knn = NearestNeighbors(n_neighbors=10)
# knn.fit(songs_normalized)
# print("Finished fitting")
# # 4. Find 10 nearest songs for a chosen song
# song_index = 0  # replace with the index of the song you want to find similar songs to
# coldplay_yellow = {"id": "4gzpq5DPGxSnKTe4SA8HAU", "popularity": 86, "duration_ms": 266773, "explicit": 0, "danceability": 0.429, "energy": 0.661, "key": 11, "loudness": -7.227, "speechiness": 0.0281, "acousticness": 0.00239, "instrumentalness": 0.000121, "liveness": 0.234, "valence": 0.285, "tempo": 173.372}
# coldplay_yellow = pd.DataFrame([coldplay_yellow]).set_index("id")
# coldplay_yellow = scaler.transform(coldplay_yellow)
# distances, indices = knn.kneighbors(coldplay_yellow)

# # Print the recommended songs
# for i in indices[0]:
#     print(songs.iloc[i]['name'])
# recommended_songs = songs.iloc[indices[0]]
