from typing import Annotated, Literal, Self

from fastmcp.tools import tool
from pydantic import Field, StringConstraints

from app.classes import CollectionConfigDict, RootConfigDict
from app.constants import MAX_STRING_LENGTH
from app.utils.config_handler import ConfigHandler


class CollectionManager:
    @classmethod
    @tool()
    def list_collections(cls: type[Self]) -> list[str]:
        """List the available collections you can use. Start with this command."""
        root_config: RootConfigDict = ConfigHandler.get_root_config()
        return root_config.collections

    @classmethod
    @tool()
    def add_collection(
        cls: type[Self],
        collection_name: Annotated[
            str, StringConstraints(to_lower=True), Field(min_length=3, max_length=MAX_STRING_LENGTH)
        ],
        embedding_provider: Literal["ollama"],
        embedding_model: Annotated[str, Field(min_length=3, max_length=MAX_STRING_LENGTH)],
    ) -> CollectionConfigDict:
        """Create a new collection.

        Args:
            collection_name: Lowercase name for the new collection.
                You will need this name to retrieve data from it.
            embedding_provider: Any supported Embedding provider. (Currently ollama supported only)
            embedding_model: Any Embedding model supported by the embedding_provider
        """
        root_config: RootConfigDict = ConfigHandler.get_root_config()
        collections = root_config.collections
        if collection_name in collections:
            exception_message = f"{collection_name} already exists in the collecions list."
            raise ValueError(exception_message)

        collections.append(collection_name)
        root_config.collections = collections
        ConfigHandler.set_root_config(root_config)

        collection_config = CollectionConfigDict(embedding_provider=embedding_provider, embedding_model=embedding_model)
        ConfigHandler.set_collection_config(collection_name=collection_name, config=collection_config)

        return collection_config
