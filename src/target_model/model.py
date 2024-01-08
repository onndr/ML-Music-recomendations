import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model

from src.data_prep.datapreprocessing import load_data


class TargetModel:
    def __init__(self, embedding_dim=50, gru_units=100, optimizer='adam', loss='sparse_categorical_crossentropy'):
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.optimizer = optimizer
        self.loss = loss
        self.model = None

    def build_model(self, input_dim, output_dim, max_sequence_length):
        self.model = Sequential([
            Embedding(input_dim=input_dim, output_dim=self.embedding_dim, input_length=max_sequence_length, mask_zero=True),
            GRU(self.gru_units, activation='tanh', input_shape=(max_sequence_length, self.embedding_dim), unroll=True, return_sequences=True),
            Dense(output_dim, activation='softmax')
        ])
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def train(self, input_sequences, target_items, epochs=50, batch_size=3):
        self.model.fit(input_sequences, target_items, epochs=epochs, batch_size=batch_size)

    def predict(self, input_sequence):
        predictions = self.model.predict(input_sequence)
        predicted_items = [pred[::-1] for pred in np.argsort(predictions[0])]
        predicted_items_last_step = predicted_items[-1]
        return predicted_items_last_step

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


# Generate sample data (replace this with your actual data)
sessions = [[1, 1, 2, 6],
            [1, 2, 3, 4],
            [2, 1, 2],
            [5, 1, 2, 6],
            [5, 1, 2, 6],
            [5, 1, 2, 6],
            [5, 1, 2, 6],
            [5, 1, 2, 6],
            [5, 1, 2, 6],
            [5, 1, 2, 6]]

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
model = TargetModel(embedding_dim=50, gru_units=100, optimizer='adam', loss='sparse_categorical_crossentropy')
model.build_model(input_dim=len(unique_items)+1, output_dim=len(unique_items)+1, max_sequence_length=max_sequence_length)

# Train the model
model.train(padded_input_sequences, padded_target_items, epochs=50, batch_size=3)

# Make predictions for a new session
new_session = np.array([[1, 3]])
padded_new_session = tf.keras.preprocessing.sequence.pad_sequences(new_session, padding='post', maxlen=max_sequence_length)
predicted_items = model.predict(padded_new_session)

# Convert item indices to actual items
recommended_items = [index_to_item[index] for index in predicted_items if index != padding_value]

print("Recommended Items:", recommended_items)
