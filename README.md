# FilmFinder  
Deep learning model to find movie genres using Pre-trained Bert


| Model Pre-trained | Loss Fuction | Average AUC. | Average F1 |
|-------------------|-------------------|----------|----------|
| bert-base-uncased | BCEWithLogitsLoss |  0.8551  |  0.5268  |
| bert-base-uncased | BalancedLogLoss   |  0.8624  |  0.5422  |


# How to setup  

## Install
```
pip install -r requirements.txt
pip install -e .
```

# Train new model

download dataset from 
    https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv

to 
    filmfinder/src/data/movies_metadata.csv


### Setup training params
filmfinder/src/train_config.yaml  

### train
```
python filmfinder/src/train.py
```

### eval the results and get nesseary files for API
exp_id is folder name under filmfinder/experiments
```
python filmfinder/src/eval.py {exp_id}
```

# To Run API  

download model from 
    https://drive.google.com/file/d/1vW7QLO6e2-h0gBII9yjPE15RNzYXpV8M/view?usp=sharing

extract zipfile to
    filmfinder/experiments/N_202306132218/best_model.ckpt
    filmfinder/experiments/N_202306132218/eval_data.pkl
    filmfinder/experiments/N_202306132218/label_data.json

### or use any new train model under filmfinder/experiments

# RUN API 
have 2 options

## 1. Docker

Build docker image
```
docker build -t film .
```
Run docker
```
docker run -p 8000:8000 film
```

## 2. Install locally

from root directory
```
cd filmfinder/api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Test request to api using curl
```
curl -X POST "http://localhost:8000/" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"overview\":\"A movie about penguins in Antarctica building a spaceship to go to Mars.\"}"
```

### Test request to api using python requests
```
import requests

url = "http://localhost:8000/"
text = "A movie about penguins in Antarctica building a spaceship to go to Mars."
res = requests.post(url, json={"overview": text})
print (res.json())
```

### exmaple return
```
[{'genre': 'Family', 'confidence': 0.7903}, {'genre': 'Animation', 'confidence': 0.6667}, {'genre': 'Comedy', 'confidence': 0.6007}, {'genre': 'Adventure', 'confidence': 0.5714}, {'genre': 'Fantasy', 'confidence': 0.4171}]
```

# Run unit test
```
cd filmfinder/api
python -m unittest discover -s tests
```