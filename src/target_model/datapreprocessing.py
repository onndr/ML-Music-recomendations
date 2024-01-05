import pandas as pd


class SessionsDataPreprocessor:
    def __init__(self):
        self._sessions_df = None

    def load_data(self, path):
        self._sessions_df = pd.read_json(path, lines=True)

    def preprocess(self):
        self._sessions_df = self._sessions_df[self._sessions_df['event_type'] == 'play']
        self._sessions_df = self._sessions_df.drop('event_type', axis=1)
        self._sessions_df = self._sessions_df.groupby('session_id').agg(list).reset_index()
        self._sessions_df['user_id'] = self._sessions_df['user_id'].apply(lambda x: x[0])
        self._sessions_df = self._sessions_df.drop(columns=['session_id', 'timestamp'])
        self._sessions_df = self._sessions_df.rename(columns={'track_id': 'tracks_ids'})

    def save_preprocessed_data(self, path):
        self._sessions_df.to_json(path, orient='records', lines=True)


if __name__ == '__main__':
    preprocessor = SessionsDataPreprocessor()
    preprocessor.load_data('../../data/sessions.jsonl')
    preprocessor.preprocess()
    preprocessor.save_preprocessed_data('../../data/sessions_preprocessed.jsonl')
