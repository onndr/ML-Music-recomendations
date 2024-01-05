from src.data_prep.datapreprocessing import load_data, preprocess_sessions, preprocess_tracks, preprocess_users, save_data
import pandas as pd


class MostPopularTracksModel:
    def __init__(self):
        self._dataset = None

    def train(self, tracks_dataset):
        self._dataset = tracks_dataset
        self._dataset.sort_values(by='popularity', ascending=False, inplace=True)

    def predict(self, n_tracks=10):
        return self._dataset['id'][:n_tracks].tolist()

    def save(self, path):
        if self._dataset is not None:
            self._dataset.to_json(path, orient='records', lines=True)


class MostListenedTracksModel:
    def __init__(self):
        self.genre_track_counts = None
        self.merged = None

    def train(self, users_dataset, sessions_dataset):
        users_df = users_dataset.explode('favourite_genres')
        sessions_df = sessions_dataset
        # Merge datasets
        self.merged = pd.merge(sessions_df, users_df, on='user_id')

        # Count the number of times each track is listened to by users with each favourite genre
        self.genre_track_counts = self.merged.groupby(['favourite_genres', 'track_id']).size()

    def predict(self, favourite_genre, n_tracks=10):
        # Get the tracks with the highest count for the given genre
        top_tracks = self.genre_track_counts[favourite_genre].nlargest(n_tracks).index.tolist()
        return top_tracks

    def save(self, path_merged, path_genre_track_counts, n_best_tracks=10):
        if self.merged.empty is not None and not self.merged.empty:
            self.merged.to_json(path_merged, orient='records', lines=True)
        if self.genre_track_counts is not None and not self.genre_track_counts.empty:
            # Create an empty DataFrame to store the results
            results = pd.DataFrame(columns=['genre', 'track_id', 'count'])

            # For each genre, get the top tracks and add them to the results DataFrame
            for genre in self.genre_track_counts.index.get_level_values('favourite_genres').unique():
                top_tracks = self.genre_track_counts[genre].nlargest(n_best_tracks)
                for track_id, count in top_tracks.items():
                    new_row = pd.DataFrame({'genre': [genre], 'track_id': [track_id], 'count': [count]})
                    results = pd.concat([results, new_row], ignore_index=True)

            # Save the results DataFrame to JSON
            results.to_json(path_genre_track_counts, orient='records', lines=True)

    def load(self, merged_path):
        self.merged = pd.read_json(merged_path, lines=True)
        self.genre_track_counts = self.merged.groupby(['favourite_genres', 'track_id']).size()


def main():
    # Load data
    # sessions_df = load_data('../../data/sessions.jsonl')
    # sessions_df = sessions_df[sessions_df['event_type'] == 'play'][['user_id', 'track_id', 'session_id']]
    # users_df = load_data('../../data/users.jsonl')[['user_id', 'favourite_genres']]

    # Train model
    most_listened_tracks_model = MostListenedTracksModel()
    most_listened_tracks_model.load('../../data/merged.jsonl')

    # Save model
    most_listened_tracks_model.save('../../data/merged.jsonl', '../../data/most_listened_tracks_model.jsonl')

    # Predict
    favourite_genre = 'rock'
    n_tracks = 10
    most_listened_tracks = most_listened_tracks_model.predict(favourite_genre, n_tracks)
    print(f'Most listened tracks for genre {favourite_genre}: {most_listened_tracks}')


if __name__ == '__main__':
    main()
