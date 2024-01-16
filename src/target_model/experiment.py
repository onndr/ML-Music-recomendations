from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from src.data_prep.datapreprocessing import load_data

import pickle

df = load_data("../../data/listened_tracks.jsonl")

# Assuming you have a DataFrame 'df' with columns: 'user_id', 'song_id', 'rating'
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(df[['user_id', 'track_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD for matrix factorization
model = SVD()
model.fit(trainset)

# Predictions for the test set
predictions = model.test(testset)

# Then compute RMSE or MAE
accuracy.rmse(predictions)
accuracy.mae(predictions)

# save predictions to a file
import json

# Assuming 'predictions' is your list
with open('predictions.json', 'w') as f:
    json.dump(predictions, f)

# Save the trained model as a pickle string.
saved_model = pickle.dumps(model)

# Save the pickled model to a file
with open('svd_model.pkl', 'wb') as file:
    file.write(saved_model)
