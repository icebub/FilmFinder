from typing import List

from pydantic import BaseModel


class BaseRequest(BaseModel):
    overview: str


class Genre(BaseModel):
    genre: str
    confidence: float


class ResponseGenre(BaseModel):
    genres: List[Genre]
