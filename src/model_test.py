import json
import time

import pandas as pd

from src.base_model.model import MostPopularTracksModel, MostListenedTracksModel
from src.data_prep.datapreprocessing import load_data
from src.target_model.knn_model import KNNModel


class ModelTester:
    def __init__(self, model, clean_sessions_df: pd.DataFrame, complete_sessions_df):
        self.model = model
        self.clean_sessions_df = clean_sessions_df
        self.complete_sessions_df = complete_sessions_df

    def get_n_prev_songs_of_user_by_timestamp(self, user_id: int, n: int, timestamp: pd.Timestamp):
        user_songs = self.clean_sessions_df[self.clean_sessions_df['user_id'] == user_id]
        user_songs = user_songs[user_songs['timestamp'] <= timestamp]
        user_songs = user_songs.sort_values(by=['timestamp'], ascending=False)
        return list(user_songs['track_id'])[:n]

    def get_n_next_songs_of_user_by_timestamp(self, user_id: int, n: int, timestamp: pd.Timestamp):
        user_songs = self.clean_sessions_df[self.clean_sessions_df['user_id'] == user_id]
        user_songs = user_songs[user_songs['timestamp'] >= timestamp]
        user_songs = user_songs.sort_values(by=['timestamp'], ascending=True)
        return list(user_songs['track_id'])[:n]

    def percent_of_common_items(self, predicted_values, actual_values):
        # get percentage of songs that are in both lists
        set1 = set(predicted_values)
        set2 = set(actual_values)
        common_songs = set1.intersection(set2)
        n = len(common_songs)
        s = len(set1)
        if s == 0:
            return 0
        return n / s

    def test(self, user_id, timestamp, k_prev_songs=10):
        pass

    def test_by_user(self, user_id, k_prev_songs=10):
        pass

    def test_all_users(self, k_prev_songs=10):
        all_users = list(self.clean_sessions_df['user_id'].unique())
        accuracies = {}
        for user_id in all_users:
            if user_id not in accuracies:
                accuracies[user_id] = []
            else:
                accuracies[user_id].append(self.test_by_user(user_id, k_prev_songs))
        return accuracies

    def analytical_measure_not_skipped(self, user_id, recommended_songs_ids, from_timestamp, n_future_skips=20):
        # zarekomendowane utwory nie zostały pominięte przez użytkownika
        # procent rekomendowanych utworów które zostały pominięte w niedalekiej przyszłości przez danego użytkownika

        if self.complete_sessions_df is None:
            raise Exception("complete_sessions_df is None")

        user_sessions = self.complete_sessions_df[self.complete_sessions_df['user_id'] == user_id]
        skipped_songs = user_sessions[user_sessions['event_type'] == 'skip']
        skipped_songs = skipped_songs[skipped_songs['timestamp'] >= from_timestamp]
        skipped_songs = skipped_songs.sort_values(by=['timestamp'], ascending=True)[:n_future_skips]
        skipped_songs_ids = list(skipped_songs['track_id'])

        return 1 - self.percent_of_common_items(recommended_songs_ids, skipped_songs_ids)

    def analytical_measure_chosen(self, user_id, recommended_songs_ids, from_timestamp, n_future_plays=20):
        # użytkownik wybierze 1 z 10 zaproponowanych utworów
        # procent rekomendowanych utworów które zostały odsłuchane w niedalekiej przyszłości przez danego użytkownika

        played_songs_ids = self.get_n_next_songs_of_user_by_timestamp(user_id, n_future_plays, from_timestamp)
        return self.percent_of_common_items(recommended_songs_ids, played_songs_ids)


class KNNModelTester(ModelTester):
    def __init__(self, model: KNNModel, clean_sessions_df: pd.DataFrame, complete_sessions_df: pd.DataFrame = None):
        super().__init__(model, clean_sessions_df, complete_sessions_df)

    def get_songs_params_by_ids(self, songs_ids):
        return self.model.all_songs[self.model.all_songs['id'].isin(songs_ids)][self.model.attributes_list]

    def test(self, user_id, timestamp, k_prev_songs=10):
        before_songs_ids = self.get_n_prev_songs_of_user_by_timestamp(user_id, k_prev_songs, timestamp)
        after_songs_ids = self.get_n_next_songs_of_user_by_timestamp(user_id, self.model.n, timestamp)

        before_songs = self.get_songs_params_by_ids(before_songs_ids)

        input_avg_before_songs = KNNModel.avg_song(before_songs)
        predicted_songs_ids = list(self.model.predict(input_avg_before_songs)['id'])

        return self.percent_of_common_items(predicted_songs_ids, after_songs_ids)

    def test_by_user(self, user_id, k_prev_songs=10):
        user_songs = self.clean_sessions_df[self.clean_sessions_df['user_id'] == user_id]
        user_songs = user_songs.sort_values(by=['timestamp'], ascending=True)
        user_songs_ids = list(user_songs['track_id'])
        accuracies = []
        for i in range(k_prev_songs, len(user_songs_ids) - self.model.n):
            before_songs_ids = user_songs_ids[i - k_prev_songs:i]
            after_songs_ids = user_songs_ids[i:i + self.model.n]

            before_songs = self.get_songs_params_by_ids(before_songs_ids)

            input_avg_before_songs = KNNModel.avg_song(before_songs)
            predicted_songs_ids = list(self.model.predict(input_avg_before_songs)['id'])

            accuracies.append(self.percent_of_common_items(predicted_songs_ids, after_songs_ids))
        return accuracies


class MostPopularTracksModelTester(ModelTester):
    def __init__(self, model: MostPopularTracksModel, clean_sessions_df: pd.DataFrame,
                 complete_sessions_df: pd.DataFrame = None):
        super().__init__(model, clean_sessions_df, complete_sessions_df)

    def test(self, user_id, timestamp, n_songs_to_predict=10):
        after_songs_ids = self.get_n_next_songs_of_user_by_timestamp(user_id, n_songs_to_predict, timestamp)
        predicted_songs_ids = self.model.predict(n_songs_to_predict)

        return self.percent_of_common_items(predicted_songs_ids, after_songs_ids)

    def test_by_user(self, user_id, n_songs_to_predict=10):
        user_songs = self.clean_sessions_df[self.clean_sessions_df['user_id'] == user_id]
        user_songs = user_songs.sort_values(by=['timestamp'], ascending=True)
        user_songs_ids = list(user_songs['track_id'])
        accuracies = []
        predicted_songs_ids = self.model.predict(n_songs_to_predict)
        for i in range(n_songs_to_predict, len(user_songs_ids) - n_songs_to_predict):
            after_songs_ids = user_songs_ids[i:i + n_songs_to_predict]

            accuracies.append(self.percent_of_common_items(predicted_songs_ids, after_songs_ids))
        return accuracies


class MostListenedTrackModelTester(ModelTester):
    def __init__(self, model: MostListenedTracksModel, clean_sessions_df: pd.DataFrame, users_df: pd.DataFrame,
                 complete_sessions_df: pd.DataFrame = None):
        super().__init__(model, clean_sessions_df, complete_sessions_df)
        self.users_df = users_df

    def test(self, user_id, timestamp, n_songs_to_predict=10):
        after_songs_ids = self.get_n_next_songs_of_user_by_timestamp(user_id, n_songs_to_predict, timestamp)
        users_genres = self.users_df[self.users_df['user_id'] == user_id]['favourite_genres'].tolist()
        predicted_songs_ids = self.model.predict(users_genres, n_songs_to_predict)

        return self.percent_of_common_items(predicted_songs_ids, after_songs_ids)

    def test_by_user(self, user_id, n_songs_to_predict=10):
        user_songs = self.clean_sessions_df[self.clean_sessions_df['user_id'] == user_id]
        user_songs = user_songs.sort_values(by=['timestamp'], ascending=True)
        user_songs_ids = list(user_songs['track_id'])
        accuracies = []
        users_genres = self.users_df[self.users_df['user_id'] == user_id]['favourite_genres'].tolist()
        predicted_songs_ids = self.model.predict(users_genres, n_songs_to_predict)
        for i in range(n_songs_to_predict, len(user_songs_ids) - n_songs_to_predict):
            after_songs_ids = user_songs_ids[i:i + n_songs_to_predict]

            accuracies.append(self.percent_of_common_items(predicted_songs_ids, after_songs_ids))
        return accuracies


def test_mlt(user_id, n):
    saved_mlt_model_path = "base_model/saved_models/most_listened_model.jsonl"
    saved_users_path = "../data/users.jsonl"

    clean_sessions_df = load_data("../data/sessions_clean.jsonl")
    users_df = load_data(saved_users_path)

    model = MostListenedTracksModel()
    model.load(saved_mlt_model_path)

    mlttester = MostListenedTrackModelTester(model, clean_sessions_df, users_df)

    results = mlttester.test_by_user(user_id, n)
    non_zero_results = [i for i in results if i > 0]
    print("MLT result: ", len(non_zero_results) / len(results))


def test_mpt(user_id, n):
    saved_mp_model_path = "base_model/saved_models/most_popular_model.jsonl"

    clean_sessions_df = load_data("../data/sessions_clean.jsonl")
    model = MostPopularTracksModel()
    model.load(saved_mp_model_path)

    mptester = MostPopularTracksModelTester(model, clean_sessions_df, None)

    results = mptester.test_by_user(user_id, n)
    non_zero_results = [i for i in results if i > 0]
    print("MPT result: ", len(non_zero_results) / len(results))


def main_knn(user_id, n):
    all_songs_df = load_data("../data/tracks.jsonl")

    model = KNNModel()
    train_set = model.fit_data_preprocessor(all_songs_df)
    model.fit(train_set, n)

    tester = KNNModelTester(model, load_data("../data/sessions_clean.jsonl"), None)

    results = tester.test_by_user(user_id, 4)
    non_zero_results = [i for i in results if i > 0]
    print(f"KNN result: ", len(non_zero_results) / len(results))


def count_analytical_measures_and_compare(iters, n_songs_to_predict, n_future_skips, n_future_plays):
    clean_sessions_df = load_data("../data/sessions_clean.jsonl")
    complete_sessions_df = load_data("../data/sessions.jsonl")
    all_songs_df = load_data("../data/tracks.jsonl")
    users_df = load_data("../data/users.jsonl")

    times = {
        "knn_train": None,
        "knn_predict": [],
        "mpt_train": None,
        "mpt_predict": [],
        "mlt_train": None,
        "mlt_predict": []
    }

    model = KNNModel()
    train_set = model.fit_data_preprocessor(all_songs_df)
    start = time.time()
    model.fit(train_set, n_songs_to_predict)
    end = time.time()
    times["knn_train"] = "%.2f" % (end - start)
    knn_tester = KNNModelTester(model, clean_sessions_df, complete_sessions_df)

    mpt_model = MostPopularTracksModel()
    start = time.time()
    mpt_model.train(all_songs_df)
    end = time.time()
    times["mpt_train"] = "%.2f" % (end - start)
    # mpt_model.load("../base_model/saved_models/most_popular_model.jsonl")

    mlt_model = MostListenedTracksModel()
    start = time.time()
    mlt_model.train(users_df, clean_sessions_df)
    end = time.time()
    times["mlt_train"] = "%.2f" % (end - start)
    # mlt_model.load("../base_model/saved_models/most_listened_model.jsonl")

    knn_metrics = {
        "am_not_skipped": [],
        "am_chosen": []
    }
    mpt_metrics = {
        "am_not_skipped": [],
        "am_chosen": []
    }
    mlt_metrics = {
        "am_not_skipped": [],
        "am_chosen": []
    }
    for i in range(iters):
        # pick random timestamp from sessions_clean
        timestamp = clean_sessions_df.sample(1)['timestamp'].iloc[0]
        for user_id in users_df['user_id']:
            try:
                prev_songs = knn_tester.get_n_prev_songs_of_user_by_timestamp(user_id, n_songs_to_predict, timestamp)
                songs_with_params = knn_tester.get_songs_params_by_ids(prev_songs)
                avg_song = model.avg_song(songs_with_params)
                if avg_song.empty or avg_song.isnull().values.any():
                    continue

                start = time.time()
                knn_pred = list(model.predict(avg_song)['id'])
                end = time.time()
                times["knn_predict"].append(end - start)

                start = time.time()
                mpt_pred = mpt_model.predict(n_songs_to_predict)
                end = time.time()
                times["mpt_predict"].append(end - start)

                users_genres = users_df[users_df['user_id'] == user_id]['favourite_genres'].tolist()
                start = time.time()
                mlt_pred = mlt_model.predict(users_genres, n_songs_to_predict)
                end = time.time()
                times["mlt_predict"].append(end - start)

                am_not_skipped_knn = knn_tester.analytical_measure_not_skipped(user_id, knn_pred, timestamp,
                                                                               n_future_skips)
                am_not_skipped_mpt = knn_tester.analytical_measure_not_skipped(user_id, mpt_pred, timestamp,
                                                                               n_future_skips)
                am_not_skipped_mlt = knn_tester.analytical_measure_not_skipped(user_id, mlt_pred, timestamp,
                                                                               n_future_skips)

                am_chosen_knn = knn_tester.analytical_measure_chosen(user_id, knn_pred, timestamp, n_future_plays)
                am_chosen_mpt = knn_tester.analytical_measure_chosen(user_id, mpt_pred, timestamp, n_future_plays)
                am_chosen_mlt = knn_tester.analytical_measure_chosen(user_id, mlt_pred, timestamp, n_future_plays)
            except Exception as e:
                print(e)
                continue

            knn_metrics["am_not_skipped"].append(am_not_skipped_knn)
            knn_metrics["am_chosen"].append(am_chosen_knn)
            mpt_metrics["am_not_skipped"].append(am_not_skipped_mpt)
            mpt_metrics["am_chosen"].append(am_chosen_mpt)
            mlt_metrics["am_not_skipped"].append(am_not_skipped_mlt)
            mlt_metrics["am_chosen"].append(am_chosen_mlt)

    times["knn_predict"] = "%.2f" % (sum(times["knn_predict"]) / len(times["knn_predict"]))
    times["mpt_predict"] = "%.2f" % (sum(times["mpt_predict"]) / len(times["mpt_predict"]))
    times["mlt_predict"] = "%.2f" % (sum(times["mlt_predict"]) / len(times["mlt_predict"]))

    print(times)

    return knn_metrics, mpt_metrics, mlt_metrics, times


if __name__ == '__main__':
    iters = 100
    n_songs_to_predict = 10
    # those are params for analytical measures
    n_future_skips = 30
    n_future_plays = 30

    knn_metrics, mpt_metrics, mlt_metrics, times = count_analytical_measures_and_compare(iters, n_songs_to_predict,
                                                                                         n_future_skips, n_future_plays)

    # write results to files
    with open("target_model/knn_metrics.json", "w") as f:
        json.dump(knn_metrics, f)
    with open("target_model/mpt_metrics.json", "w") as f:
        json.dump(mpt_metrics, f)
    with open("target_model/mlt_metrics.json", "w") as f:
        json.dump(mlt_metrics, f)

    avg_knn_not_skipped = sum(knn_metrics["am_not_skipped"]) / len(knn_metrics["am_not_skipped"]) * 100
    avg_knn_chosen = sum(knn_metrics["am_chosen"]) / len(knn_metrics["am_chosen"]) * 100
    avg_mpt_not_skipped = sum(mpt_metrics["am_not_skipped"]) / len(mpt_metrics["am_not_skipped"]) * 100
    avg_mpt_chosen = sum(mpt_metrics["am_chosen"]) / len(mpt_metrics["am_chosen"]) * 100
    avg_mlt_not_skipped = sum(mlt_metrics["am_not_skipped"]) / len(mlt_metrics["am_not_skipped"]) * 100
    avg_mlt_chosen = sum(mlt_metrics["am_chosen"]) / len(mlt_metrics["am_chosen"]) * 100
    print("KNN KA1: ", "%.3f" % avg_knn_not_skipped, "%")
    print("KNN KA2: ", "%.3f" % avg_knn_chosen, "%")
    print("MPT KA1: ", "%.3f" % avg_mpt_not_skipped, "%")
    print("MPT KA2: ", "%.3f" % avg_mpt_chosen, "%")
    print("MLT KA1: ", "%.3f" % avg_mlt_not_skipped, "%")
    print("MLT KA2: ", "%.3f" % avg_mlt_chosen, "%")
