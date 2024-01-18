from datapreprocessing import load_data
from datetime import datetime

import pandas as pd


def avg_song(songs: pd.DataFrame):
    return songs.mean(numeric_only=True).to_frame().transpose()


def get_n_last_songs_of_user_by_timestamp(songs_df: pd.DataFrame, user_id: int, n: int, timestamp: pd.Timestamp):
    user_songs = songs_df[songs_df['user_id'] == user_id]
    user_songs = user_songs[user_songs['timestamp'] <= timestamp]
    user_songs = user_songs.sort_values(by=['timestamp'], ascending=False)
    return user_songs[['track_id']][:n]


def get_songs_by_ids(songs_df: pd.DataFrame, songs_ids: list):
    return songs_df[songs_df['id'].isin(songs_ids)]


def prepare_songs_df():
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
    return songs[songs_attrs]


def main():
    N = 10
    USER_ID = 101
    DATE = pd.Timestamp(datetime(2023, 10, 4, 2, 38, 38))
    print(DATE)
    sessions_df = load_data("data/sessions_clean.jsonl")

    last_songs = get_n_last_songs_of_user_by_timestamp(sessions_df, USER_ID, N, DATE)
    songs_corr = prepare_songs_df()
    selected_songs_df = get_songs_by_ids(songs_corr, last_songs['track_id'].tolist())

    avg = avg_song(selected_songs_df)
    print(avg)


if __name__ == '__main__':
    main()
