from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from app.constants import MAX_VARIABLE_STRING_LENGTH

CollectionName = Annotated[
    str, StringConstraints(to_lower=True), Field(min_length=3, max_length=MAX_VARIABLE_STRING_LENGTH)
]
EmbeddingProvider = Literal["ollama"]
MetadataValue = str | int | float | bool | list[str] | list[int] | list[float]
Metadata = dict[str, MetadataValue] | None
ExistingMetadata = dict[str, MetadataValue]
