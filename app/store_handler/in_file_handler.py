from pathlib import Path

import numpy as np
from ulid import ULID

from app.types import CollectionName
from app.utils.file_handler import FileHandler


class InFileStoreHandler:
    def __init__(
        self,
        collection_name: CollectionName,
    ):
        self.data_store_folder: Path = FileHandler.get_collection_data_folder(collection_name=collection_name)
        config_store_folder: Path = FileHandler.get_collection_folder(collection_name=collection_name)
        self.vector_file_path = config_store_folder / "vectors.npy"
        self.ids_file_path = config_store_folder / "ids.npy"

    def append_batch(
        self,
        embeddings: list[list[float]],
        ids: list[str],
    ):
        if len(embeddings) != len(ids):
            error_message = f"{len(embeddings)=} and {len(ids)=} are not equal."
            raise ValueError(error_message)

        # (batch_size, dim)
        new_vectors = np.asarray(embeddings, dtype=np.float32)

        if new_vectors.ndim != 2:
            error_message = f"The embeddings should be a 2D array {new_vectors.ndim}"
            raise ValueError(error_message)

        # Normalization for cosine similarity
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)

        if np.any(norms == 0):
            error_message = "Zero vector found in the batch."
            raise ValueError(error_message)

        new_vectors = new_vectors / norms

        new_ids = np.asarray(ids)

        try:
            old_vectors = np.load(self.vector_file_path)
            old_ids = np.load(self.ids_file_path)

            if old_vectors.shape[1] != new_vectors.shape[1]:
                error_message = f"Dimension mismatch: {old_vectors.shape[1]} != {new_vectors.shape[1]}"
                raise ValueError(error_message)

            vectors = np.vstack([old_vectors, new_vectors])
            all_ids = np.concatenate([old_ids, new_ids])

        except FileNotFoundError:
            vectors = new_vectors
            all_ids = new_ids

        np.save(self.vector_file_path, vectors)
        np.save(self.ids_file_path, all_ids)

    def store_values(self, stored_values: list[str]) -> list[str]:
        """Stores the files and returns their ULIDs"""
        list_of_ulids: list[str] = []
        for stored_value in stored_values:
            ulid = ULID()
            file_path = self.data_store_folder / f"{ulid}.txt"
            file_path.write_text(stored_value, encoding="utf8")
            list_of_ulids.append(str(ulid))

        return list_of_ulids
