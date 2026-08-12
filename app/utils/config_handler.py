from functools import cache, lru_cache
from typing import Self

from app.classes.base_classes import CollectionConfigDict, RootConfigDict
from app.utils.file_handler import FileHandler


class ConfigHandler:
    @classmethod
    @cache
    def get_root_config(cls: type[Self]) -> RootConfigDict:
        """We are loading the root config file."""
        root_config_file = FileHandler.get_config_location()

        if root_config_file.exists():
            return RootConfigDict.model_validate_json(json_data=root_config_file.read_text(encoding="utf8"))
        return RootConfigDict(collections=[])

    @classmethod
    def set_root_config(cls: type[Self], config: RootConfigDict) -> None:
        """We are updating the root config file."""
        root_config_file = FileHandler.get_config_location()
        root_config_file.write_text(config.model_dump_json(indent=4), encoding="utf8")
        cls.get_root_config.cache_clear()

    @classmethod
    @lru_cache
    def get_collection_config(cls: type[Self], collection_name: str) -> CollectionConfigDict:
        """We are loading the collection specific config file."""
        collection_config = FileHandler.get_collection_folder(collection_name=collection_name) / "config.json"

        if collection_config.exists():
            return CollectionConfigDict.model_validate_json(json_data=collection_config.read_text(encoding="utf8"))
        exception_message = f"file {collection_config=} is not exists"
        raise FileNotFoundError(exception_message)

    @classmethod
    def set_collection_config(cls: type[Self], collection_name: str, config: CollectionConfigDict) -> None:
        """We are updating the collection config file."""
        collection_config = FileHandler.get_collection_folder(collection_name=collection_name) / "config.json"
        collection_config.write_text(config.model_dump_json(indent=4), encoding="utf8")
        cls.get_collection_config.cache_clear()
