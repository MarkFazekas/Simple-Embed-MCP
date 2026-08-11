from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.constants import MAX_STRING_LENGTH


class RootConfigDict(BaseModel):
    collections: list[str]


class CollectionConfigDict(BaseModel):
    embedding_provider: Literal["ollama"]
    embedding_model: Annotated[str, Field(min_length=3, max_length=MAX_STRING_LENGTH)]
