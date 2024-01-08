import pandas as pd


def clean_sessions(sessions_df):
    sessions_df = sessions_df[sessions_df['event_type'] == 'play']
    sessions_df = sessions_df.drop('event_type', axis=1)
    return sessions_df


def load_data(path):
    return pd.read_json(path, lines=True)


def save_data(data: pd.DataFrame, path):
    data.to_json(path, orient='records', lines=True)


def preprocess_sessions_group_by_session_id(sessions_df):
    sessions_df = clean_sessions(sessions_df)
    # group by session_id and aggregate values in lists
    sessions_df = sessions_df.groupby('session_id').agg(list).reset_index()
    # we dont need a list of the same user_id repeated for each track_id
    sessions_df['user_id'] = sessions_df['user_id'].apply(lambda x: x[0])
    # sort by timestamp and user_id
    sessions_df = sessions_df.sort_values(by=['user_id', 'timestamp'],
                                          key=lambda col: col.str[0] if col.name == 'timestamp' else col,
                                          ascending=True)
    # rename track_id column
    sessions_df = sessions_df.rename(columns={'track_id': 'tracks_ids', 'timestamp': 'timestamps'})
    # add length of each session
    sessions_df['session_length'] = sessions_df['tracks_ids'].apply(lambda x: len(x))
    # get max session length
    max_session_length = sessions_df['session_length'].max()
    # get min session length
    min_session_length = sessions_df['session_length'].min()
    return sessions_df, max_session_length, min_session_length


def preprocess_sessions_group_by_user_id(sessions_df):
    sessions_df = clean_sessions(sessions_df)
    # sort by timestamp
    sessions_df = sessions_df.sort_values('timestamp', ascending=True)
    # group by session_id and aggregate values in lists
    sessions_df = sessions_df.groupby('user_id').agg(list).reset_index()
    # remove session_id column
    sessions_df = sessions_df.drop('session_id', axis=1)
    # rename track_id and session_id columns
    sessions_df = sessions_df.rename(columns={'track_id': 'tracks_ids', 'timestamp': 'timestamps'})
    # add length of each user history
    sessions_df['history_length'] = sessions_df['tracks_ids'].apply(lambda x: len(x))
    # get max history length
    max_history_length = sessions_df['history_length'].max()
    # get min history length
    min_history_length = sessions_df['history_length'].min()
    return sessions_df, max_history_length, min_history_length


def transform_sessions_discrete():
    df = load_data("../../data/sessions.jsonl")
    df, max_, min_ = preprocess_sessions_group_by_session_id(df)
    save_data(df, "../../data/discrete_users_sessions.jsonl")

    print("max: ", max_)
    print("min: ", min_)


def transform_sessions_continuous():
    df = load_data("../../data/sessions.jsonl")
    df, max_, min_ = preprocess_sessions_group_by_user_id(df)
    save_data(df, "../../data/full_users_history.jsonl")

    print("max: ", max_)
    print("min: ", min_)


if __name__ == '__main__':
    transform_sessions_discrete()
    # transform_sessions_continuous()
