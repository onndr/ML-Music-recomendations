from datetime import datetime
import pandas as pd

from src.data_prep.datapreprocessing import load_data
from src.target_model.knn_model import KNNModel


class KNNModelTester:
    def __init__(self, model: KNNModel, all_sessions_df: pd.DataFrame):
        self.model = model
        self.all_sessions_df = all_sessions_df

    def get_n_prev_songs_of_user_by_timestamp(self, user_id: int, n: int, timestamp: pd.Timestamp):
        user_songs = self.all_sessions_df[self.all_sessions_df['user_id'] == user_id]
        user_songs = user_songs[user_songs['timestamp'] <= timestamp]
        user_songs = user_songs.sort_values(by=['timestamp'], ascending=False)
        return list(user_songs['track_id'])[:n]

    def get_n_next_songs_of_user_by_timestamp(self, user_id: int, n: int, timestamp: pd.Timestamp):
        user_songs = self.all_sessions_df[self.all_sessions_df['user_id'] == user_id]
        user_songs = user_songs[user_songs['timestamp'] >= timestamp]
        user_songs = user_songs.sort_values(by=['timestamp'], ascending=True)
        return list(user_songs['track_id'])[:n]

    def get_songs_params_by_ids(self, songs_ids):
        return self.model.all_songs[self.model.all_songs['id'].isin(songs_ids)][self.model.attributes_list]

    def accuracy(self, predicted_values, actual_values):
        # get percentage of songs that are in both lists
        set1 = set(predicted_values)
        set2 = set(actual_values)
        common_songs = set1.intersection(set2)
        n = len(common_songs)
        s = len(set1)
        return n/s

    def test(self, user_id, timestamp, k_prev_songs=10):
        before_songs_ids = self.get_n_prev_songs_of_user_by_timestamp(user_id, k_prev_songs, timestamp)
        after_songs_ids = self.get_n_next_songs_of_user_by_timestamp(user_id, self.model.n, timestamp)

        before_songs = self.get_songs_params_by_ids(before_songs_ids)

        input_avg_before_songs = KNNModel.avg_song(before_songs)
        predicted_songs_ids = list(self.model.predict(input_avg_before_songs)['id'])

        return self.accuracy(predicted_songs_ids, after_songs_ids)


def main():
    N = 10
    USER_ID = 101
    DATE = pd.Timestamp(datetime(2022, 7, 13, 1, 31, 0))

    all_songs_df = load_data("../../data/tracks.jsonl")

    model = KNNModel()
    train_set = model.fit_data_preprocessor(all_songs_df)
    model.fit(train_set, N)

    tester = KNNModelTester(model, load_data("../../data/sessions_clean.jsonl"))
    print("test accuracy: ", tester.test(USER_ID, DATE))


if __name__ == '__main__':
    main()
