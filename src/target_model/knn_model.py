import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def load_data(path: str):
    return pd.read_json(path, lines=True)


# 1. Load song data
songs = load_data('data/tracks.jsonl')
songs_attrs = [
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
songs_corr = songs[songs_attrs]
songs_corr = songs_corr.set_index("id")
scaler = StandardScaler().fit(songs_corr)
songs_normalized = scaler.transform(songs_corr)
print("Start fitting")
# 3. Create kNN model and fit it to the data
knn = NearestNeighbors(n_neighbors=10)
knn.fit(songs_normalized)
print("Finished fitting")
# 4. Find 10 nearest songs for a chosen song
song_index = 0  # replace with the index of the song you want to find similar songs to
coldplay_yellow = {"id": "4gzpq5DPGxSnKTe4SA8HAU", "popularity": 86, "duration_ms": 266773, "explicit": 0, "danceability": 0.429, "energy": 0.661, "key": 11, "loudness": -7.227, "speechiness": 0.0281, "acousticness": 0.00239, "instrumentalness": 0.000121, "liveness": 0.234, "valence": 0.285, "tempo": 173.372}
coldplay_yellow = pd.DataFrame([coldplay_yellow]).set_index("id")
coldplay_yellow = scaler.transform(coldplay_yellow)
distances, indices = knn.kneighbors(coldplay_yellow)

# Print the recommended songs
for i in indices[0]:
    print(songs.iloc[i]['name'])
recommended_songs = songs.iloc[indices[0]]
