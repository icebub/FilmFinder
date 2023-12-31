FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ADD . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install -e .

WORKDIR /app/filmfinder/api

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0 0.0", "--port", "8000"]