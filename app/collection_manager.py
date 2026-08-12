from pathlib import Path
from typing import Annotated, Literal, Self

from fastmcp.tools import tool
from pydantic import Field

from app.classes import (
    BatchCollectionAdditionOperation,
    CollectionAdditionOperation,
    CollectionConfigDict,
    RootConfigDict,
)
from app.constants import MAX_VARIABLE_STRING_LENGTH
from app.embeding_handler import EmbeddingHandler
from app.store_handler.in_file_handler import InFileSearchResult, InFileStoreHandler
from app.types import CollectionName, EmbeddingProvider
from app.utils.config_handler import ConfigHandler


class CollectionManager:
    @classmethod
    @tool()
    def list_collections(cls: type[Self]) -> dict[str, list[str] | dict[str, CollectionConfigDict]]:
        """List the available collections you can use. Start with this command."""
        root_config: RootConfigDict = ConfigHandler.get_root_config()
        collections_info: dict[str, CollectionConfigDict] = {
            collection_name: ConfigHandler.get_collection_config(collection_name)
            for collection_name in root_config.collections
        }
        return_dict: dict[str, list[str] | dict[str, CollectionConfigDict]] = {
            "collections": root_config.collections,
            "collections_info": collections_info,
        }
        return return_dict

    @classmethod
    @tool()
    def add_collection(
        cls: type[Self],
        collection_name: CollectionName,
        collection_storage: Literal["in_file"],
        embedding_provider: EmbeddingProvider,
        embedding_model: Annotated[str, Field(min_length=3, max_length=MAX_VARIABLE_STRING_LENGTH)],
    ) -> CollectionConfigDict:
        """Create a new collection.

        Args:
            collection_name: Lowercase name for the new collection.
                You will need this name to retrieve data from it.
            collection_storage: Any of the supported storage providers. (Currently in_file supported only)
            embedding_provider: Any of the supported embedding providers. (Currently ollama supported only)
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

        collection_config = CollectionConfigDict(
            embedding_provider=embedding_provider,
            collection_storage=collection_storage,
            embedding_model=embedding_model,
        )
        ConfigHandler.set_collection_config(collection_name=collection_name, config=collection_config)

        return collection_config

    @classmethod
    @tool(timeout=30)
    def add_value_to_collection(
        cls: type[Self],
        collection_name: CollectionName,
        stored_value: Annotated[str, Field(min_length=3)],
        embeddable_key: str | None = None,
    ) -> list[str]:
        """Adds a new value to an existing collection.

        Args:
            collection_name: Name of the collection.
            stored_value: The string value which we will store and return. (BM25 search will use it.)
            embeddable_key: The embedding vector will be calculated based on this if present.
                If embeddable_key is not provided, stored_value will be used.
                (Only the embedding vector of this value will be stored.)
        """
        addition_operation = CollectionAdditionOperation(stored_value=stored_value, embeddable_key=embeddable_key)
        return cls.add_values_to_collection(collection_name=collection_name, addition_operations=[addition_operation])

    @classmethod
    @tool(timeout=600)
    def batch_add_values_to_collection(
        cls: type[Self],
        collection_name: CollectionName,
        batch_file_path: Path,
    ) -> list[str]:
        """Adds a new values to an existing collection.

        Args:
            collection_name: Name of the collection.
            batch_file_path: The json file path where we have a batch CollectionAdditionOperation structure:
                {"batch": [CollectionAdditionOperation]}
                CollectionAdditionOperation:
                - stored_value: The string value which we will store and return. (BM25 search will use it.)
                - embeddable_key: The embedding vector will be calculated based on this if present.
                       If embeddable_key is not provided, stored_value will be used.
                       (Only the embedding vector of this value will be stored.)
        """
        if not batch_file_path.exists():
            error_message = f"{batch_file_path=} is not exists"
            raise ValueError(error_message)

        addition_operations_text = batch_file_path.read_text(encoding="utf8")
        addition_operations = BatchCollectionAdditionOperation.model_validate_json(addition_operations_text)
        return cls.add_values_to_collection(
            collection_name=collection_name, addition_operations=addition_operations.batch
        )

    @classmethod
    def add_values_to_collection(
        cls: type[Self], collection_name: CollectionName, addition_operations: list[CollectionAdditionOperation]
    ) -> list[str]:
        """Adds values to an existing collection in batch.

        Args:
            collection_name: Name of the collection.
            addition_operations: List of CollectionAdditionOperations to execute.
        """
        collection_config: CollectionConfigDict = ConfigHandler.get_collection_config(collection_name=collection_name)

        embeddable_keys = [
            addition_operation.embeddable_key or addition_operation.stored_value
            for addition_operation in addition_operations
        ]

        embedding_vectors: list[list[float]] = EmbeddingHandler.generate_embeddings(
            embedding_provider=collection_config.embedding_provider,
            embedding_model=collection_config.embedding_model,
            input_texts=embeddable_keys,
        )

        match collection_config.collection_storage:
            case "in_file":
                store_handler = InFileStoreHandler(collection_name=collection_name)
                ids: list[str] = store_handler.store_key_values(addition_operations=addition_operations)
                store_handler.append_batch(embeddings=embedding_vectors, ids=ids)
                return ids
        return []

    @classmethod
    @tool(timeout=60)
    def search_in_collection_embedding(
        cls: type[Self],
        collection_name: CollectionName,
        search_key: str,
        number_of_return_values: int = 10,
    ) -> list[InFileSearchResult]:
        """Returns the top number_of_return_values matching search_key in the collection.
        It uses only cosine similarity search.

        Args:
            collection_name: Name of the collection.
            search_key: The string value which we will use to search in the collection's embedding vectors.
            number_of_return_values: The number of values to return.
        """
        collection_config: CollectionConfigDict = ConfigHandler.get_collection_config(collection_name=collection_name)

        embedding_vectors: list[list[float]] = EmbeddingHandler.generate_embeddings(
            embedding_provider=collection_config.embedding_provider,
            embedding_model=collection_config.embedding_model,
            input_texts=[search_key],
        )
        embedding_vector = embedding_vectors[0]

        match collection_config.collection_storage:
            case "in_file":
                store_handler = InFileStoreHandler(collection_name=collection_name)
                result = store_handler.search_embedding(query_embedding=embedding_vector, k=number_of_return_values)
                return result

    @classmethod
    @tool(timeout=60)
    def search_in_collection_keys_bm25(
        cls: type[Self],
        collection_name: CollectionName,
        search_key: str,
        number_of_return_values: int = 10,
    ) -> list[InFileSearchResult]:
        """Returns the top number_of_return_values matching search_key in the collection keys.
        It uses only BM25 search.

        Args:
            collection_name: Name of the collection.
            search_key: The string value which we will use to search in the collection's keys.
            number_of_return_values: The number of values to return.
        """
        collection_config: CollectionConfigDict = ConfigHandler.get_collection_config(collection_name=collection_name)

        match collection_config.collection_storage:
            case "in_file":
                store_handler = InFileStoreHandler(collection_name=collection_name)
                result = store_handler.search_bm25_by_key(query=search_key, top_k=number_of_return_values)
                return result

    @classmethod
    @tool(timeout=60)
    def search_in_collection_values_bm25(
        cls: type[Self],
        collection_name: CollectionName,
        search_key: str,
        number_of_return_values: int = 10,
    ) -> list[InFileSearchResult]:
        """Returns the top number_of_return_values matching search_key in the collection values.
        It uses only BM25 search.

        Args:
            collection_name: Name of the collection.
            search_key: The string value which we will use to search in the collection's values.
            number_of_return_values: The number of values to return.
        """
        collection_config: CollectionConfigDict = ConfigHandler.get_collection_config(collection_name=collection_name)

        match collection_config.collection_storage:
            case "in_file":
                store_handler = InFileStoreHandler(collection_name=collection_name)
                result = store_handler.search_bm25_by_value(query=search_key, top_k=number_of_return_values)
                return result
