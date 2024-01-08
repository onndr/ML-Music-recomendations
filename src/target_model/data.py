
import numpy as np
from sklearn.calibration import LabelEncoder
from src.data_prep.datapreprocessing import load_data
from tensorflow.keras.preprocessing.sequence import pad_sequences


SESSIONS_PATH = "../../data/discrete_users_sessions.jsonl"
# there are 562222 sessions in total
df = load_data(SESSIONS_PATH)
max_seq_length = df['session_length'].max()
print("max: ", max_seq_length)

# Extract the sessions and user data
sessions = df['tracks_ids'].tolist()
user_data = df['user_id'].tolist()
tracks = load_data("../../data/tracks.jsonl")['id'].tolist()

# Flatten the list of sessions and fit the encoder
encoder = LabelEncoder()
encoder.fit(tracks)

# Transform the sessions
sessions = [list(encoder.transform(session)) for session in sessions]

# Create input sequences and target values
input_sequences = []
target_values = []
user_data_expanded = []
for user_id, session in zip(user_data, sessions):
    for i in range(1, len(session)):
        input_sequences.append(session[:i])
        target_values.append(session[i])
        user_data_expanded.append(user_id)

# Convert expanded user data to numpy array
user_data = np.array(user_data_expanded)

# Pad sequences to a fixed length
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')
target_values = np.array(target_values)

# Save the arrays to .npy files
np.save('input_sequences.npy', input_sequences)
np.save('target_values.npy', target_values)
np.save('user_data.npy', user_data)
