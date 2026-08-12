import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
from ulid import ULID

from app.types import CollectionName
from app.utils.file_handler import FileHandler


class InFileSearchResult(TypedDict):
    index: int
    id: str
    score: float
    result_text: str


class BM25:
    def __init__(self, documents: list[str], document_ids: list[str], k1=1.5, b=0.75):
        self.term_frequency_sat = k1
        self.document_length_norm = b

        self.document_ids = document_ids
        self.document_texts = documents
        self.documents = [self.tokenize(doc) for doc in documents]
        self.number_of_documents = len(self.documents)

        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / self.number_of_documents if self.number_of_documents else 0

        self.term_frequencies = []
        self.document_frequencies = Counter()

        for doc in self.documents:
            tf = Counter(doc)
            self.term_frequencies.append(tf)

            for term in tf:
                self.document_frequencies[term] += 1

        self.idf = {}
        for term, df in self.document_frequencies.items():
            # Robertson/Sparck Jones-style IDF with BM25+ stabilization
            self.idf[term] = math.log(1 + (self.number_of_documents - df + 0.5) / (df + 0.5))

    @staticmethod
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    def score(self, query, doc_index):
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

    def search(self, query, top_k=5) -> list[InFileSearchResult]:
        results = [(i, self.score(query, i)) for i in range(self.number_of_documents)]

        results.sort(key=lambda x: x[1], reverse=True)

        return [
            InFileSearchResult(
                index=index,
                id=self.document_ids[index],
                score=score,
                result_text=self.document_texts[index],
            )
            for index, score in results[:top_k]
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
        self.stored_file_path: Callable[[str | ULID], Path] = lambda ulid: self.data_store_folder / f"{ulid}.txt"

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
        self.get_bm25_obj.cache_clear()

    @classmethod
    @lru_cache
    def get_np_vectors(cls, vector_file_path: Path, ids_file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        vectors: np.ndarray = np.load(vector_file_path)
        ids: np.ndarray = np.load(ids_file_path)
        return vectors, ids

    @classmethod
    @lru_cache
    def get_bm25_obj(cls, data_store_folder: Path, ids_file_path: Path) -> BM25:
        ids: np.ndarray = np.load(ids_file_path)
        document_ids = [str(ulid) for ulid in ids]
        document_texts = [(data_store_folder / f"{ulid}.txt").read_text(encoding="utf8") for ulid in document_ids]
        return BM25(documents=document_texts, document_ids=document_ids)

    def store_values(self, stored_values: list[str]) -> list[str]:
        """Stores the files and returns their ULIDs"""
        list_of_ulids: list[str] = []
        for stored_value in stored_values:
            ulid = ULID()
            file_path = self.stored_file_path(ulid)
            file_path.write_text(stored_value, encoding="utf8")
            list_of_ulids.append(str(ulid))

        return list_of_ulids

    def search_embedding(
        self,
        query_embedding: list[float],
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

        k = min(k, len(scores))

        # We don't order the whole array, we just select the top k.
        indexes = np.argpartition(scores, -k)[-k:]

        # We sort within the top k.
        indexes = indexes[np.argsort(scores[indexes])[::-1]]

        return [
            InFileSearchResult(
                index=int(i),
                id=str(ids[i]),
                score=float(scores[i]),
                result_text=self.stored_file_path(str(ids[i])).read_text(encoding="utf8"),
            )
            for i in indexes
        ]

    def search_bm25(self, query: str, top_k: int = 5) -> list[InFileSearchResult]:
        bm25 = self.get_bm25_obj(data_store_folder=self.data_store_folder, ids_file_path=self.ids_file_path)
        return bm25.search(query, top_k=top_k)
