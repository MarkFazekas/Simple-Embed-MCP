import json
import math
import re
from collections import Counter
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
from ulid import ULID

from app.classes.base_classes import CollectionAdditionOperation
from app.classes.metadata_filter import MetaDataFilter
from app.types import CollectionName, Metadata
from app.utils.file_handler import FileHandler


class InFileSearchResult(TypedDict):
    index: int
    id: str
    score: float
    result_text: str
    metadata: Metadata


class BM25:
    def __init__(
        self,
        searchable_documents: list[str],
        result_documents: list[str],
        metadata_list: list[Metadata],
        document_ids: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.term_frequency_sat = k1
        self.document_length_norm = b

        self.document_ids = document_ids
        self.result_documents = result_documents
        self.metadata_list = metadata_list
        self.searchable_documents = searchable_documents
        self.searchable_words = [self.tokenize(doc) for doc in searchable_documents]
        self.number_of_documents = len(self.searchable_words)

        self.doc_lengths = [len(doc) for doc in self.searchable_words]
        self.avg_doc_length = sum(self.doc_lengths) / self.number_of_documents if self.number_of_documents else 0

        self.term_frequencies = []
        self.document_frequencies: Counter = Counter()

        for doc in self.searchable_words:
            tf = Counter(doc)
            self.term_frequencies.append(tf)

            for term in tf:
                self.document_frequencies[term] += 1

        self.idf = {}
        for term, df in self.document_frequencies.items():
            # Robertson/Sparck Jones-style IDF with BM25+ stabilization
            self.idf[term] = math.log(1 + (self.number_of_documents - df + 0.5) / (df + 0.5))

    @staticmethod
    def tokenize(text: str):
        return re.findall(r"\b\w+\b", text.lower())

    def score(self, query: str, doc_index: int) -> float:
        query_terms = self.tokenize(query)

        tf = self.term_frequencies[doc_index]
        doc_length = self.doc_lengths[doc_index]

        score = 0.0

        for term in query_terms:
            if term not in self.idf:
                continue

            frequency = tf.get(term, 0)
            if frequency == 0:
                continue

            numerator = frequency * (self.term_frequency_sat + 1)

            denominator = frequency + self.term_frequency_sat * (
                1 - self.document_length_norm + self.document_length_norm * doc_length / self.avg_doc_length
            )

            score += self.idf[term] * numerator / denominator

        return score

    def search(self, query: str, metadata_filter: MetaDataFilter, top_k: int) -> list[InFileSearchResult]:
        results = [(i, self.score(query, i)) for i in range(self.number_of_documents)]

        matching = [
            (i, s) for i, s in results if s > 0 and metadata_filter.calculate_condition_block(self.metadata_list[i])
        ]
        matching.sort(key=lambda x: x[1], reverse=True)

        return [
            InFileSearchResult(
                index=index,
                id=self.document_ids[index],
                score=score,
                result_text=self.result_documents[index],
                metadata=self.metadata_list[index],
            )
            for index, score in matching[:top_k]
            if score > 0
        ]


class InFileStoreHandler:
    def __init__(
        self,
        collection_name: CollectionName,
    ):
        self.data_store_folder: Path = FileHandler.get_collection_data_folder(collection_name=collection_name)
        config_store_folder: Path = FileHandler.get_collection_folder(collection_name=collection_name)
        self.vector_file_path = config_store_folder / "vectors.npy"
        self.ids_file_path = config_store_folder / "ids.npy"
        self.stored_value_file_path: Callable[[str | ULID], Path] = partial(
            self.get_stored_value_file_path, self.data_store_folder
        )
        self.stored_key_file_path: Callable[[str | ULID], Path] = partial(
            self.get_stored_key_file_path, self.data_store_folder
        )
        self.stored_meta_file_path: Callable[[str | ULID], Path] = partial(
            self.get_stored_meta_file_path, self.data_store_folder
        )

    @classmethod
    def get_stored_value_file_path(cls, data_store_folder: Path, ulid: str | ULID) -> Path:
        return data_store_folder / f"{ulid}_value.txt"

    @classmethod
    def get_stored_key_file_path(cls, data_store_folder: Path, ulid: str | ULID) -> Path:
        return data_store_folder / f"{ulid}_key.txt"

    @classmethod
    def get_stored_meta_file_path(cls, data_store_folder: Path, ulid: str | ULID) -> Path:
        return data_store_folder / f"{ulid}_meta.json"

    def append_batch(
        self,
        embeddings: list[list[float]],
        ids: list[str],
    ) -> None:
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
            old_vectors, old_ids = self.get_np_vectors(self.vector_file_path, self.ids_file_path)

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
        self.get_np_vectors.cache_clear()
        self.get_bm25_key_only_filter.cache_clear()
        self.get_bm25_value_only_filter.cache_clear()

    @classmethod
    @lru_cache
    def get_np_vectors(cls, vector_file_path: Path, ids_file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        vectors: np.ndarray = np.load(vector_file_path)
        ids: np.ndarray = np.load(ids_file_path)
        return vectors, ids

    @classmethod
    @lru_cache
    def get_bm25_key_only_filter(cls, data_store_folder: Path, ids_file_path: Path) -> BM25:
        ids: np.ndarray = np.load(ids_file_path)
        searchable_documents = []
        result_documents = []
        document_ids = []
        metadata_list: list[Metadata] = []
        for np_ulid in ids:
            ulid = str(np_ulid)
            key_path = cls.get_stored_key_file_path(data_store_folder, ulid)
            if key_path.exists():
                document_ids.append(ulid)
                key_content = key_path.read_text(encoding="utf8")
                searchable_documents.append(key_content)
                value_path = cls.get_stored_value_file_path(data_store_folder, ulid)
                value_content = value_path.read_text(encoding="utf8")
                result_documents.append(value_content)
                metadata_list.append(cls.get_metadata_by_ulid(data_store_folder, ulid))

        return BM25(
            searchable_documents=searchable_documents,
            document_ids=document_ids,
            metadata_list=metadata_list,
            result_documents=result_documents,
        )

    @classmethod
    @lru_cache
    def get_bm25_value_only_filter(cls, data_store_folder: Path, ids_file_path: Path) -> BM25:
        ids: np.ndarray = np.load(ids_file_path)
        result_documents = []
        document_ids = []
        metadata_list = []
        for np_ulid in ids:
            ulid = str(np_ulid)
            document_ids.append(ulid)
            value_path = cls.get_stored_value_file_path(data_store_folder, ulid)
            value_content = value_path.read_text(encoding="utf8")
            result_documents.append(value_content)
            metadata_list.append(cls.get_metadata_by_ulid(data_store_folder, ulid))

        return BM25(
            searchable_documents=result_documents,
            document_ids=document_ids,
            result_documents=result_documents,
            metadata_list=metadata_list,
        )

    def store_key_values(self, addition_operations: list[CollectionAdditionOperation]) -> list[str]:
        """Stores the files and returns their ULIDs"""
        list_of_ulids: list[str] = []
        for addition_operation in addition_operations:
            ulid = ULID()
            file_path = self.stored_value_file_path(ulid)
            file_path.write_text(addition_operation.stored_value, encoding="utf8")
            list_of_ulids.append(str(ulid))
            if addition_operation.embeddable_key:
                file_path = self.stored_key_file_path(ulid)
                file_path.write_text(addition_operation.embeddable_key, encoding="utf8")
            if addition_operation.metadata:
                file_path = self.stored_meta_file_path(ulid)
                file_path.write_text(json.dumps(addition_operation.metadata, indent=4), encoding="utf8")

        return list_of_ulids

    @classmethod
    @lru_cache(maxsize=None)
    def get_metadata_by_ulid(cls, data_store_folder: Path, ulid: str | ULID) -> Metadata:
        file_path = cls.get_stored_meta_file_path(data_store_folder, ulid)
        if file_path.exists():
            return json.loads(file_path.read_text(encoding="utf8"))
        return None

    def search_embedding(
        self,
        query_embedding: list[float],
        metadata_filter: MetaDataFilter,
        k: int = 5,
    ) -> list[InFileSearchResult]:
        vectors, ids = self.get_np_vectors(self.vector_file_path, self.ids_file_path)

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim != 1:
            error_message = f"The query embedding should be a 1D vector. {query.ndim=}"
            raise ValueError(error_message)

        if query.shape[0] != vectors.shape[1]:
            error_message = f"Dimension mismatch: {query.shape[0]} != {vectors.shape[1]}"
            raise ValueError(error_message)

        norm = np.linalg.norm(query)

        if norm == 0.0:
            error_message = "The query can not be a zero vector."
            raise ValueError(error_message)

        query = query / norm

        # cosine similarity with all vectors
        scores = vectors @ query

        # Metadata filter
        mask = np.array(
            [
                metadata_filter.calculate_condition_block(
                    metadata=self.get_metadata_by_ulid(
                        data_store_folder=self.data_store_folder,
                        ulid=str(ulid),
                    )
                )
                for ulid in ids
            ],
            dtype=bool,
        )

        # Exclude vectors that don't match the metadata filter
        scores[~mask] = -np.inf

        matching_count = int(mask.sum())

        if matching_count == 0:
            return []

        k = min(k, matching_count)

        # We don't order the whole array, we just select the top k.
        indexes = np.argpartition(scores, -k)[-k:]

        # We sort within the top k.
        indexes = indexes[np.argsort(scores[indexes])[::-1]]

        return [
            InFileSearchResult(
                index=int(i),
                id=str(ids[i]),
                score=float(scores[i]),
                result_text=self.stored_value_file_path(ulid=str(ids[i])).read_text(encoding="utf8"),
                metadata=self.get_metadata_by_ulid(
                    data_store_folder=self.data_store_folder,
                    ulid=str(ids[i]),
                ),
            )
            for i in indexes
        ]

    def search_bm25_by_value(self, query: str, metadata_filter: MetaDataFilter, top_k: int) -> list[InFileSearchResult]:
        bm25 = self.get_bm25_value_only_filter(
            data_store_folder=self.data_store_folder, ids_file_path=self.ids_file_path
        )
        return bm25.search(query, metadata_filter=metadata_filter, top_k=top_k)

    def search_bm25_by_key(self, query: str, metadata_filter: MetaDataFilter, top_k: int) -> list[InFileSearchResult]:
        bm25 = self.get_bm25_key_only_filter(data_store_folder=self.data_store_folder, ids_file_path=self.ids_file_path)
        return bm25.search(query, metadata_filter=metadata_filter, top_k=top_k)
