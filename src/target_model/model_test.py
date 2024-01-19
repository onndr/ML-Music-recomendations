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
        if n/s > 0:
            pass
        return n/s

    def test(self, user_id, timestamp, k_prev_songs=10):
        before_songs_ids = self.get_n_prev_songs_of_user_by_timestamp(user_id, k_prev_songs, timestamp)
        after_songs_ids = self.get_n_next_songs_of_user_by_timestamp(user_id, self.model.n, timestamp)

        before_songs = self.get_songs_params_by_ids(before_songs_ids)

        input_avg_before_songs = KNNModel.avg_song(before_songs)
        predicted_songs_ids = list(self.model.predict(input_avg_before_songs)['id'])

        return self.accuracy(predicted_songs_ids, after_songs_ids)

    def test_by_user(self, user_id, k_prev_songs=10):
        user_songs = self.all_sessions_df[self.all_sessions_df['user_id'] == user_id]
        user_songs = user_songs.sort_values(by=['timestamp'], ascending=True)
        user_songs_ids = list(user_songs['track_id'])
        user_songs_timestamps = list(user_songs['timestamp'])
        accuracies = []
        for i in range(k_prev_songs, len(user_songs_ids)-self.model.n):
            before_songs_ids = user_songs_ids[i-k_prev_songs:i]
            after_songs_ids = user_songs_ids[i:i+self.model.n]
            before_songs = self.get_songs_params_by_ids(before_songs_ids)
            input_avg_before_songs = KNNModel.avg_song(before_songs)
            predicted_songs_ids = list(self.model.predict(input_avg_before_songs)['id'])
            accuracies.append(self.accuracy(predicted_songs_ids, after_songs_ids))
        return accuracies

    def test_all_users(self, k_prev_songs=10):
        all_users = list(self.all_sessions_df['user_id'].unique())
        accuracies = []
        for user_id in all_users:
            accuracies.append(self.test_by_user(user_id, k_prev_songs))
        return accuracies


def main():
    N = 10
    USER_ID = 1000
    DATE = pd.Timestamp(datetime(2022, 7, 13, 1, 31, 0))

    all_songs_df = load_data("../../data/tracks.jsonl")

    model = KNNModel()
    train_set = model.fit_data_preprocessor(all_songs_df)
    model.fit(train_set, N)

    tester = KNNModelTester(model, load_data("../../data/sessions_clean.jsonl"))
    preds = tester.test_by_user(USER_ID, 20)
    print(f"how good is the model for the user with id {USER_ID}: ", len([i for i in preds if i > 0])/len(preds))


if __name__ == '__main__':
    main()
