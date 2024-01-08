import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, concatenate, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from src.data_prep.datapreprocessing import load_data

# # Example session data (songs listened to in sessions)
# sessions = [
#     [1, 3, 2, 4, 1],
#     [2, 5, 1, 3],
#     [3, 4, 2, 1, 5, 2],
# ]

# # Example user data (user IDs corresponding to the sessions)
# user_data = np.array([1, 2, 3])  # Assuming user IDs correspond to sessions in order

# # Additional session data (songs listened to in sessions)
# sessions += [
#     [2, 1, 4, 3, 5],
#     [1, 2, 3, 4, 5, 1, 2],
#     [5, 4, 3, 2, 1],
#     [1, 2, 1, 2, 1, 2, 1],
# ]


# # Additional user data (user IDs corresponding to the sessions)
# user_data = np.concatenate([user_data, np.array([1, 1, 2, 3])])  # Assuming user IDs correspond to sessions in order

# # Additional session data (songs listened to in sessions)
# sessions += [
#     [2, 1, 4, 3, 5],
#     [1, 2, 3, 4, 5, 1, 2],
#     [5, 4, 3, 2, 1],
#     [1, 2, 1, 2, 1, 2, 1],
#     [3, 2, 5, 1, 4, 3, 2],
#     [4, 5, 1, 2, 3, 4, 5],
#     [1, 3, 2, 4, 1, 3, 2],
#     [2, 4, 1, 3, 2, 4, 1],
# ]

# # Additional user data (user IDs corresponding to the sessions)
# user_data = np.concatenate([user_data, np.array([2, 2, 1, 1, 1, 2, 3, 3])])  # Assuming user IDs correspond to sessions in order

# # Additional session data (songs listened to in sessions)
# sessions += [
#     [2, 1, 4, 3, 5],
#     [1, 2, 3, 4, 5, 1, 2],
#     [5, 4, 3, 2, 1],
#     [1, 2, 1, 2, 1, 2, 1],
#     [3, 2, 5, 1, 4, 3, 2],
#     [4, 5, 1, 2, 3, 4, 5],
#     [1, 3, 2, 4, 1, 3, 2],
#     [2, 4, 1, 3, 2, 4, 1],
# ]

# # Additional user data (user IDs corresponding to the sessions)
# user_data = np.concatenate([user_data, np.array([1, 1, 2, 2, 3, 3, 1, 2])])  # Assuming user IDs correspond to sessions in order

# # Additional session data (songs listened to in sessions)
# sessions += [
#     [5, 4, 3, 2, 1],
#     [1, 2, 3, 4, 5],
#     [2, 3, 4, 5, 1],
#     [3, 4, 5, 1, 2],
#     [4, 5, 1, 2, 3],
# ]

# # Additional user data (user IDs corresponding to the sessions)
# user_data = np.concatenate([user_data, np.array([1, 2, 3, 1, 2])])  # Assuming user IDs correspond to sessions in order

# # Create vocabulary and convert sessions to sequences of equal length
# vocab_size = len(set(np.concatenate(sessions))) + 1  # Add 1 to the maximum song ID
# num_users = len(set(user_data)) + 1  # Add 1 to the maximum user ID
# max_seq_length = max(len(session) for session in sessions)

# # Create input sequences, their respective targets, and corresponding user data
# input_sequences = []
# target_values = []
# user_data_expanded = []
# for user_id, session in zip(user_data, sessions):
#     for i in range(1, len(session)):
#         input_sequences.append(session[:i])
#         target_values.append(session[i])
#         user_data_expanded.append(user_id)

# # Convert expanded user data to numpy array
# user_data = np.array(user_data_expanded)

# # Pad sequences to a fixed length
# input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')
# target_values = np.array(target_values)

# # Ensure alignment of data lengths
# assert len(input_sequences) == len(target_values) == len(user_data)

# # Split data into train and test sets
# X_train, X_test, y_train, y_test, user_train, user_test = train_test_split(
#     input_sequences, target_values, user_data, test_size=0.2, random_state=42
# )

# Assume df is your DataFrame and it has columns 'tracks_ids' and 'user_id' which contain the sessions and user data

# SESSIONS_PATH = "../../data/discrete_users_sessions.jsonl"
# # there are 562222 sessions in total
# df = load_data(SESSIONS_PATH)[0:1000]
# max_seq_length = df['session_length'].max()
# print("max: ", max_seq_length)

# # Extract the sessions and user data
# sessions = df['tracks_ids'].tolist()
# user_data = df['user_id'].tolist()
tracks = load_data("../../data/tracks.jsonl")['id'].tolist()
# print("tracks: ", tracks)

# Flatten the list of sessions and fit the encoder
encoder = LabelEncoder()
encoder.fit(tracks)

# # Transform the sessions
# sessions = [list(encoder.transform(session)) for session in sessions]

# # Create input sequences and target values
# input_sequences = []
# target_values = []
# user_data_expanded = []
# for user_id, session in zip(user_data, sessions):
#     for i in range(1, len(session)):
#         input_sequences.append(session[:i])
#         target_values.append(session[i])
#         user_data_expanded.append(user_id)

# # Convert expanded user data to numpy array
# user_data = np.array(user_data_expanded)

# # Pad sequences to a fixed length
# input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')
# target_values = np.array(target_values)

# # Save the arrays to .npy files
# np.save('input_sequences.npy', input_sequences)
# np.save('target_values.npy', target_values)
# np.save('user_data.npy', user_data)

# Load the arrays from .npy files
input_sequences = np.load('input_sequences.npy')
target_values = np.load('target_values.npy')
user_data = np.load('user_data.npy')

max_seq_length = input_sequences.shape[1]

# Ensure alignment of data lengths
assert len(input_sequences) == len(target_values) == len(user_data)

# Split data into train and test sets
X_train, X_test, y_train, y_test, user_train, user_test = train_test_split(input_sequences, target_values, user_data, test_size=0.2, shuffle=False)

# Print out the test set
print("Test set users:")
print(user_test)

print("Test set sessions:")
print(X_test)

# Define vocabulary sizes and embedding dimensions
# vocab_size = len(set(np.concatenate(sessions))) + 1
vocab_size = len(tracks) + 1
num_users = np.max(user_data) + 1  # Add 1 to the maximum user ID
embedding_dim = 100

# User input and embedding layers
user_input = Input(shape=(1,))
user_embedding = Embedding(num_users, embedding_dim, input_length=1)(user_input)
user_embedding = Flatten()(user_embedding)

# Sessions input and embedding layers
sessions_input = Input(shape=(max_seq_length,))
sessions_embedding = Embedding(vocab_size, embedding_dim, input_length=max_seq_length)(sessions_input)
sessions_gru = GRU(100)(sessions_embedding)

# Merge user and session embeddings
merged = concatenate([user_embedding, sessions_gru])

# Additional dense layers for further processing
merged = Dense(100, activation='relu')(merged)
output = Dense(vocab_size, activation='softmax')(merged)

# Create the final model
model = Model(inputs=[user_input, sessions_input], outputs=output)

# Compile and train the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit([user_train, X_train], y_train, epochs=10, batch_size=3200, validation_split=0.1)

# Evaluate the model on test data
loss, accuracy = model.evaluate([user_test, X_test], y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

test_index = 3

# Make a prediction for a test case
test_case_user = np.array([user_test[test_index]])  # Use the user from the test set
test_case_session = np.array([X_test[test_index]])  # Use the session from the test set

# Print out the test case
print("Test case user:")
print(test_case_user)

print("Test case session:")
print(test_case_session)

prediction = model.predict([test_case_user, test_case_session])

# The prediction is a probability distribution over the vocabulary
# To get the ID of the most likely next song, find the index of the maximum value
predicted_song_id = np.argmax(prediction)

print(f'Predicted next song ID for the test case: {predicted_song_id}')
print(f'Actual next song ID for the test case: {y_test[test_index]}')

# Save the model
model.save('model.keras')

# Get the original song ID
original_song_id = encoder.inverse_transform([predicted_song_id])
real_song_id = encoder.inverse_transform([y_test[test_index]])

# Get the user ID
user_id = user_test[test_index]

print(f"Real song ID: {real_song_id[0]}, Predicted song ID: {original_song_id[0]} User ID: {user_id}")

# load tracks data
TRACKS_PATH = "../../data/tracks.jsonl"
tracks_df = load_data(TRACKS_PATH)

# Get the predicted track name
predicted_track_name = tracks_df[tracks_df['id'] == original_song_id[0]]['name'].values[0]
real_track_name = tracks_df[tracks_df['id'] == real_song_id[0]]['name'].values[0]

print(f"Predicted track name: {predicted_track_name}, Real track name: {real_track_name}")


# Write out the prediction to a file
with open('prediction.txt', 'w') as f:
    f.write("Prediction:\n")
    for song_id, probability in enumerate(prediction[0]):
        real_song_id = encoder.inverse_transform([song_id])[0]
        song_name = tracks_df[tracks_df['id'] == real_song_id]['name'].values[0]
        f.write(f"Song ID: {real_song_id}, Song name: {song_name.encode()}, Probability: {round(probability, 3)}\n")
