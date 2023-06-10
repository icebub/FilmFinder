import os

from datasets.MovieGenres import MovieGenres
from models.BaseModel import BaseModel
from transformers import BertTokenizer

model = BaseModel(num_classes=32, freeze_bert=False)

abs_folder = os.path.dirname(os.path.abspath(__file__))
dataset = MovieGenres(f"{abs_folder}/data/movies_metadata.csv")
x, y = dataset[0]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentence = x
tokens = tokenizer.tokenize(sentence)

tokens = ["[CLS]"] + tokens + ["[SEP]"]

# Convert tokens to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Print the token IDs

print(tokens)
print(token_ids)
