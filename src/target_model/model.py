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
item_to_index = {item: i for i, item in enumerate(unique_items)}
index_to_item = {i: item for i, item in enumerate(unique_items)}

sequences = [[item_to_index[item] for item in session] for session in sessions]

# Create input sequences and target items
input_sequences = [seq[:-1] for seq in sequences]
target_items = [seq[1:] for seq in sequences]

max_sequence_length = max(len(seq) for seq in input_sequences)
padded_input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, padding='post')
padded_target_items = tf.keras.preprocessing.sequence.pad_sequences(target_items, padding='post')

# Build the GRU4Rec model
embedding_dim = 50

model = Sequential([
    Embedding(input_dim=len(unique_items), output_dim=embedding_dim, input_length=max_sequence_length),
    GRU(100, activation='tanh', input_shape=(max_sequence_length, embedding_dim), unroll=True, return_sequences=True),
    Dense(len(unique_items), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_input_sequences, padded_target_items, epochs=10, batch_size=2)

# Make predictions for a new session
new_session = np.array([[1, 3, 4]])
padded_new_session = tf.keras.preprocessing.sequence.pad_sequences(new_session, padding='post', maxlen=max_sequence_length)
predictions = model.predict(padded_new_session)

# Convert predictions to item indices
predicted_items = [np.argmax(pred) for pred in predictions[0]]

# Convert item indices to actual items
recommended_items = [index_to_item[index] for index in predicted_items]

print("Recommended Items:", recommended_items)