from src.data_prep.datapreprocessing import load_data
import pandas as pd


class MostPopularTracksModel:
    def __init__(self):
        self._dataset = None

    def train(self, tracks_dataset, n_to_hold=10):
        self._dataset = tracks_dataset
        self._dataset.sort_values(by='popularity', ascending=False, inplace=True)
        self._dataset = self._dataset[:n_to_hold]

    def predict(self, n_tracks=10):
        return self._dataset['id'][:n_tracks].tolist()

    def save(self, path):
        if self._dataset is not None:
            self._dataset.to_json(path, orient='records', lines=True)

    def load(self, path):
        self._dataset = load_data(path)


class MostListenedTracksModel:
    def __init__(self):
        self.counts_df = None

    def train(self, users_dataset, sessions_dataset, tracks_for_genre=10):
        # unpack favourite_genres list into separate rows
        users_df = users_dataset.explode('favourite_genres')

        # basically a join on user_id
        merged = pd.merge(sessions_dataset, users_df, on='user_id')

        # group by favourite_genres and track_id and count the number of occurrences
        self.counts_df = merged.groupby(['favourite_genres', 'track_id'])
        self.counts_df.as_index = False
        self.counts_df = self.counts_df.size()

        # sort by favourite_genres and count
        self.counts_df = self.counts_df.sort_values(['favourite_genres', 'size'], ascending=[True, False])
        self.counts_df = self.counts_df.groupby('favourite_genres').head(tracks_for_genre)

        # rename columns
        self.counts_df = self.counts_df.rename(columns={'favourite_genres': 'genre', 'size': 'count'})

    def predict(self, genres, n_tracks=10):
        # get the tracks with the highest count for the given genres
        top_tracks = self.counts_df.loc[self.counts_df['genre'].isin(genres)].nlargest(n_tracks, columns='count')
        return top_tracks

    def save(self, path_genre_track_counts):
        if self.counts_df is not None:
            # save the df to jsonl
            self.counts_df.to_json(path_genre_track_counts, orient='records', lines=True)

    def load(self, genre_track_counts_path):
        # load the data from jsonl
        self.counts_df = pd.read_json(genre_track_counts_path, lines=True)


def test_most_popular_tracks_model_train(tracks_path, save_model_path, n_mp_tracks, n_tracks_to_predict):
    # Load data
    tracks_df = load_data(tracks_path)[['id', 'name', 'popularity']]

    # Train model
    most_popular_tracks_model = MostPopularTracksModel()
    most_popular_tracks_model.train(tracks_df, n_mp_tracks)
    most_popular_tracks_model.save(save_model_path)

    # Predict
    most_popular_tracks = most_popular_tracks_model.predict(n_tracks_to_predict)
    print(f'Most popular tracks: {most_popular_tracks}')


def test_most_popular_tracks_model_load(saved_model_path, n_tracks_to_predict):
    # Load model
    most_popular_tracks_model = MostPopularTracksModel()
    most_popular_tracks_model.load(saved_model_path)

    # Predict
    most_popular_tracks = most_popular_tracks_model.predict(n_tracks_to_predict)
    print(f'Most popular tracks: {most_popular_tracks}')


def test_most_listened_tracks_model_train(sessions_path, clean_sessions_path, users_path,
                                          save_model_path, n_tracks_per_genre, genres):
    # Clean data
    # clean_sessions(sessions_path, clean_sessions_path)

    # Load data
    sessions_df = load_data(clean_sessions_path)[['user_id', 'track_id']]
    users_df = load_data(users_path)[['user_id', 'favourite_genres']]

    # Train model
    most_listened_tracks_model = MostListenedTracksModel()
    most_listened_tracks_model.train(users_df, sessions_df, n_tracks_per_genre)
    most_listened_tracks_model.save(save_model_path)

    # Predict
    most_listened_tracks = most_listened_tracks_model.predict(genres)
    print(f'Most listened tracks for genres {genres}: {most_listened_tracks}')


def test_most_listened_tracks_model_load(saved_model_path, genres):
    # Load model
    most_listened_tracks_model = MostListenedTracksModel()
    most_listened_tracks_model.load(saved_model_path)

    # Predict
    most_listened_tracks = most_listened_tracks_model.predict(genres)
    print(f'Most listened tracks for genres {genres}: {most_listened_tracks}')


def main():
    SESSIONS_PATH = '../../data/sessions.jsonl'
    CLEAN_SESSIONS_PATH = '../../data/sessions_clean.jsonl'
    TRACKS_PATH = '../../data/tracks.jsonl'
    USERS_PATH = '../../data/users.jsonl'
    MOST_POPULAR_MODEL_PATH = './saved_models/most_popular_model.jsonl'
    MOST_LISTENED_MODEL_PATH = './saved_models/most_listened_model.jsonl'

    N_TRACKS_TO_PREDICT = 10
    N_POPULAR_TRACKS_TO_SAVE = 100
    N_TRACKS_PER_GENRE = 10
    GENRES = ['rock', 'pop punk']

    test_most_popular_tracks_model_train(TRACKS_PATH, MOST_POPULAR_MODEL_PATH, N_POPULAR_TRACKS_TO_SAVE, N_TRACKS_TO_PREDICT)
    test_most_popular_tracks_model_load(MOST_POPULAR_MODEL_PATH, N_TRACKS_TO_PREDICT)
    test_most_listened_tracks_model_train(SESSIONS_PATH, CLEAN_SESSIONS_PATH, USERS_PATH, MOST_LISTENED_MODEL_PATH,
                                          N_TRACKS_PER_GENRE, GENRES)
    test_most_listened_tracks_model_load(MOST_LISTENED_MODEL_PATH, GENRES)


if __name__ == '__main__':
    main()
