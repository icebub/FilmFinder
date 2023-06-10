from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieGenres(Dataset):
    def __init__(self, file_path):
        metadata = pd.read_csv(file_path, low_memory=False)
        self.overviews = metadata["overview"].values
        genres = metadata["genres"].values

        self.genre_count = self.genres_count(genres)
        self.mapping, self.reverse_mapping = self.genres_mapping(self.genre_count)
        self.label = self.create_label(genres, self.mapping)

    def genres_count(self, genres):
        genre_count = defaultdict(int)
        for data in genres:
            names = [x["name"] for x in eval(data)]
            for name in names:
                genre_count[name] += 1
        return genre_count

    def genres_mapping(self, genre_count):
        genres = [(k, v) for k, v in genre_count.items()]
        genres = sorted(genres, key=lambda x: x[1], reverse=True)
        mapping = {k: i for i, (k, v) in enumerate(genres)}
        reverse_mapping = {i: k for k, i in mapping.items()}
        return mapping, reverse_mapping

    def create_label(self, genres, mapping):
        label_size = len(mapping)
        labels = []
        for data in genres:
            now_label = [0] * label_size
            names = [x["name"] for x in eval(data)]
            for name in names:
                now_label[mapping[name]] = 1
            labels.append(now_label)
        return labels

    def __getitem__(self, index):
        x = self.overviews[index]
        y = self.label[index]

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
    print("lebel_size: ", len(dataset.mapping))
