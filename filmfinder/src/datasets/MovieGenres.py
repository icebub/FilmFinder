from collections import defaultdict

import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset


class MovieGenres(Dataset):
    def __init__(self, file_path, tokenizer, max_len=512):
        metadata = pd.read_csv(file_path, low_memory=False)
        metadata = metadata.dropna(subset=["overview"])

        overviews = metadata["overview"].values
        genres = metadata["genres"].values

        self.genre_count = self.genres_count(genres)
        self.mapping, self.reverse_mapping = self.genres_mapping(self.genre_count)
        self.label = self.create_label(genres, self.mapping)

        self.train = self.preprocess(overviews, tokenizer, max_len)

    def preprocess(self, sentences, tokenizer, max_len=512):
        print("Preprocessing...")
        tokenized_sentences = []
        for sentence in tqdm.tqdm(sentences):
            tokens = tokenizer.tokenize(sentence)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_ids = token_ids[:max_len]  # limit to max_len
            padding_length = max_len - len(token_ids)
            token_ids += [0] * padding_length  # pad
            tokenized_sentences.append(token_ids)

        return torch.tensor(tokenized_sentences)

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
        return torch.tensor(labels)

    def __getitem__(self, index):
        x = self.train[index]
        y = self.label[index]

        return x, y

    def __len__(self):
        return len(self.overviews)


if __name__ == "__main__":
    import os

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    abs_folder = os.path.dirname(os.path.abspath(__file__))
    dataset = MovieGenres(f"{abs_folder}/../data/movies_metadata.csv", tokenizer)
    x, y = dataset[0]
    print(x)
    print(y)
    print("lebel_size: ", len(dataset.mapping))
