import requests

from app.types import EmbeddingProvider


class EmbeddingHandler:
    @classmethod
    def generate_embeddings(
        cls, embedding_provider: EmbeddingProvider, embedding_model: str, input_texts: list[str]
    ) -> list[list[float]]:
        """Generates embeddings with the provider."""

        match embedding_provider:
            case "ollama":
                url = "http://localhost:11434/api/embed"
                input_data = {"model": embedding_model, "input": input_texts}
                response = requests.post(url=url, json=input_data, timeout=600)
                result = response.json()
                if error := result.get("error"):
                    error_message = f"Embedding generation failed with ollama provider: {error}"
                    raise ValueError(error_message)

                embeddings: list[list[float]] = result.get("embeddings")
                return embeddings

        error_message = f"{embedding_provider=} not supported"
        raise ValueError(error_message)
