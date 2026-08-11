from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.constants import MAX_VARIABLE_STRING_LENGTH
from app.types import EmbeddingProvider


class RootConfigDict(BaseModel):
    collections: list[str]


class CollectionConfigDict(BaseModel):
    embedding_provider: EmbeddingProvider
    collection_storage: Literal["in_file"]
    embedding_model: Annotated[str, Field(min_length=3, max_length=MAX_VARIABLE_STRING_LENGTH)]


class CollectionAdditionOperation(BaseModel):
    stored_value: Annotated[str, Field(min_length=3)]
    embeddable_key: str | None = None


class BatchCollectionAdditionOperation(BaseModel):
    batch: list[CollectionAdditionOperation]
