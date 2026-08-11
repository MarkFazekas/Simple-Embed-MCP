from pathlib import Path
from typing import Self


class FileHandler:
    @classmethod
    def get_collection_folder(cls: type[Self], collection_name: str) -> Path:
        """Determine the collection specific folder location."""
        return cls._ensure_folder(cls._get_collections_folder() / collection_name)

    @classmethod
    def get_collection_data_folder(cls: type[Self], collection_name: str) -> Path:
        """Determine the collection specific folder location."""
        return cls._ensure_folder(cls.get_collection_folder(collection_name=collection_name) / "data")

    @classmethod
    def get_config_location(cls: type[Self]) -> Path:
        """Determine the Server specific config file location."""
        return cls._get_root_folder() / "config.json"

    @classmethod
    def _ensure_folder(cls: type[Self], folder_path: Path) -> Path:
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    @classmethod
    def _get_root_folder(cls: type[Self]) -> Path:
        """Determine the Simple Embed MCP folder root location."""
        return cls._ensure_folder(Path.cwd() / ".simple_embed_mcp")

    @classmethod
    def _get_collections_folder(cls: type[Self]) -> Path:
        """Determine the collections folder location."""
        return cls._ensure_folder(cls._get_root_folder() / "collections")
