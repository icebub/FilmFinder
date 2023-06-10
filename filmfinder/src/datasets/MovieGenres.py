from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieGenres(Dataset):
    def __init__(self, file_path):
        metadata = pd.read_csv(file_path, low_memory=False)
        self.overviews = metadata["overview"].values
        self.genres = metadata["genres"].values
        self.genre_count = self.get_genres_count(self.genres)

    def get_genres_count(self, genres):
        genre_count = defaultdict(int)
        for data in genres:
            names = [x["name"] for x in eval(data)]
            for name in names:
                genre_count[name] += 1
        return genre_count

    def __getitem__(self, index):
        x = self.overviews[index]
        y = self.genres[index]

        return x, y

    def __len__(self):
        return len(self.overviews)


if __name__ == "__main__":
    import os

    abs_folder = os.path.dirname(os.path.abspath(__file__))

    dataset = MovieGenres(f"{abs_folder}/../data/movies_metadata.csv")

    x, y = dataset[0]
    print(x)
    print(y)
