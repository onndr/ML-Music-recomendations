import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

# Generate sample data (replace this with your actual data)
sessions = [[1, 2, 3, 4],
            [2, 3, 5],
            [1, 4, 5, 6],
            [3, 5, 6, 7]]

unique_items = list(set(item for session in sessions for item in session))
item_to_index = {item: i for i, item in enumerate(unique_items, 1)}
index_to_item = {i: item for i, item in enumerate(unique_items, 1)}

sequences = [[item_to_index[item] for item in session] for session in sessions]

# Create input sequences and target items
input_sequences = [seq[:-1] for seq in sequences]
target_items = [seq[1:] for seq in sequences]

padding_value = 0

max_sequence_length = max(len(seq) for seq in input_sequences)
padded_input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, padding='post', value=padding_value)
padded_target_items = tf.keras.preprocessing.sequence.pad_sequences(target_items, padding='post', value=padding_value)

# Build the GRU4Rec model
embedding_dim = 50

model = Sequential([
    Embedding(input_dim=len(unique_items)+1, output_dim=embedding_dim, input_length=max_sequence_length, mask_zero=True),
    GRU(100, activation='tanh', input_shape=(max_sequence_length, embedding_dim), unroll=True, return_sequences=True),
    Dense(len(unique_items)+1, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
# TODO: CHANGE TARGET ITEMS TO PADDED TARGET ITEMS
model.fit(padded_input_sequences, np.array(padded_target_items), epochs=50, batch_size=3)

# Make predictions for a new session
new_session = np.array([[1, 5]])
padded_new_session = tf.keras.preprocessing.sequence.pad_sequences(new_session, padding='post', maxlen=max_sequence_length)
predictions = model.predict(padded_new_session)
print("Padded input session: ", padded_new_session)
# TODO: CHANGE PREDICTED ITEMS TO LIST OF PREDICTED ITEMS
# Convert predictions to item indices
# TODO: CHOOSE CORRECT SET OF PREDICTIONS
predicted_items = [pred[::-1] for pred in np.argsort(predictions[0])[:padded_new_session.shape[1]]]

predicted_items_last_step = predicted_items[-1]

# Convert item indices to actual items
# recommended_items = [[index_to_item[index] for index in pred_arr if index != padding_value] for pred_arr in predicted_items]
recommended_items = [index_to_item[index] for index in predicted_items_last_step if index != padding_value]

print("Recommended Items:", recommended_items)
