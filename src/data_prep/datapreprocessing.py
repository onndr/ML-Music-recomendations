import pandas as pd


def load_data(path):
    return pd.read_json(path, lines=True)


def save_data(data: pd.DataFrame, path):
    data.to_json(path, orient='records', lines=True)


def preprocess_sessions(sessions_df):
    sessions_df = sessions_df[sessions_df['event_type'] == 'play']
    sessions_df = sessions_df.drop('event_type', axis=1)
    sessions_df = sessions_df.groupby('session_id').agg(list).reset_index()
    sessions_df['user_id'] = sessions_df['user_id'].apply(lambda x: x[0])
    sessions_df = sessions_df.rename(columns={'track_id': 'tracks_ids'})
    return sessions_df


def preprocess_tracks(tracks_df):
    tracks_df = tracks_df[['id', 'name', 'popularity']]
    return tracks_df


def preprocess_users(users_df):
    users_df = users_df[['user_id', 'favourite_genres']]
    return users_df
