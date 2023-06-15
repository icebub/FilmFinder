import torch
from api_core import load_config, load_model, predict
from fastapi import FastAPI
from schema import BaseRequest, ResponseGenre

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global model, tokenizer, reverse_mapping, thresholds, f1_mappping, device

    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Server is running on device: ", device)

    exp_id = config["exp_id"]
    model, tokenizer, reverse_mapping, thresholds, f1_mappping = load_model(
        exp_id, device
    )

    sample_text = "Miles Morales catapults across the Multiverse, where he encounters a team of Spider-People charged with protecting its very existence. When the heroes clash on how to handle a new threat, Miles must redefine what it means to be a hero."
    return_list = predict(
        sample_text, model, tokenizer, reverse_mapping, thresholds, f1_mappping, device
    )
    print("Sample text: ", sample_text)
    print("Predicted labels: ", return_list)


@app.post("/overview")
async def process_req(request: BaseRequest, response_model=ResponseGenre):
    return_list = predict(
        request.text, model, tokenizer, reverse_mapping, thresholds, f1_mappping, device
    )
    return return_list
