import torch
from api_core import load_model, predict
from fastapi import FastAPI
from pydantic import BaseModel


class BaseRequest(BaseModel):
    text: str


app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Server is running on device: ", device)

exp_id = "N_202306132218"
model, tokenizer, reverse_mapping, thresholds = load_model(exp_id, device)

sample_text = "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their differences."
return_list = predict(
    sample_text, model, tokenizer, reverse_mapping, thresholds, device
)
print("Sample text: ", sample_text)
print("Predicted labels: ", return_list)


@app.get("/predict")
async def process_req(request: BaseRequest):
    text = request.text

    return {"text": text}
