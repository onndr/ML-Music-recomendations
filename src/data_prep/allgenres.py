import os
import datapreprocessing as dp


print(os.getcwd())
data = dp.load_data("data/artists.jsonl")


genres = []

for i in range(data.shape[0]):
    for genre in data.iloc[i]['genres']:
        genres.append(genre)

genres = list(set(genres))
print(len(genres))