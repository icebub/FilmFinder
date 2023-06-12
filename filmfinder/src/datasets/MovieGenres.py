from collections import defaultdict

import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.float)


class MovieGenres:
    def __init__(self, file_path):
        metadata = pd.read_csv(file_path, low_memory=False)
        metadata = metadata.dropna(subset=["overview"])
        # drop row original_language != en
        metadata = metadata[metadata["original_language"] == "en"]
        # drop row that genres is empty
        # metadata = metadata[metadata["genres"] != "[]"]

        self.overviews = metadata["overview"].values
        genres = metadata["genres"].values

        self.genre_count = self.genres_count(genres)
        self.mapping, self.reverse_mapping = self.genres_mapping(self.genre_count)

        self.label = self.create_label(genres, self.mapping)

    def get_dataset(self):
        return self.overviews, self.label

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
