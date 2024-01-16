from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from src.data_prep.datapreprocessing import load_data, save_data

import pickle


def prep_data():
    # Load data
    df = load_data("../../data/sessions_clean.jsonl")
    df = df[["user_id", "track_id"]]

    # Drop duplicates
    df = df.drop_duplicates()

    # add rating column
    df["rating"] = 1

    save_data(df, "../../data/listened_tracks.jsonl")
    return df


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

# If you have a binary rating system
correct_predictions = sum([1 for prediction in predictions if (prediction.r_ui > 0.5 and prediction.est > 0.5) or (
            prediction.r_ui <= 0.5 and prediction.est <= 0.5)])
accuracy_percentage = (correct_predictions / len(predictions)) * 100
print(f'Accuracy: {accuracy_percentage}%')

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
