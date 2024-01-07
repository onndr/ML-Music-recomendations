import pandas as pd


def clean_sessions(sessions_path, sessions_clean_path):
    sessions_df = load_data(sessions_path)
    sessions_df = sessions_df[sessions_df['event_type'] == 'play']
    sessions_df = sessions_df.drop('event_type', axis=1)
    sessions_df.to_json(sessions_clean_path, orient='records', lines=True)


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
