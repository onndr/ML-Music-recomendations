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


def main():
    N = 10
    USER_ID = 101
    DATE = pd.Timestamp(datetime(2023, 10, 4, 2, 38, 38))
    print(DATE)
    df = load_data("data/sessions_clean.jsonl")

    last_songs = get_n_last_songs_of_user_by_timestamp(df, USER_ID, N, DATE)

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
    merge = pd.merge(last_songs, songs_corr, left_on='track_id', right_on='id')
    avg = avg_song(merge)
    print(avg)


if __name__ == '__main__':
    main()
