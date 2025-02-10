import contextlib
import io
import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

from .base_storage import BaseStorage

from chromadb.api import ClientAPI

from swarm.utilities import EmbeddingConfigurator
from swarm.util import PATHS


@contextlib.contextmanager
def suppress_logging(
    logger_name="chromadb.segment.impl.vector.local_persistent_hnsw",
    level=logging.ERROR,
):
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.suppress(UserWarning),
    ):
        yield
    logger.setLevel(original_level)


class RAGStorage(BaseStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    app: ClientAPI | None = None

    def __init__(
        self, type, allow_reset=True, embedder_config=None
    ):
        '''
        Args:
            type: str - the type of storage to use
            allow_reset: bool - whether to allow resetting the storage
            embedder_config: dict - configuration for the embedder
            e.g.
                {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small",
                        "api_key": "your_openai_api_key"
                    }
                }

        '''
        self.storage_file_name = PATHS.SHORT_TERM_STORAGE

        self.type = type
        self.embedder_config = embedder_config
        self.allow_reset = allow_reset
        self._initialize_app()

    def _set_embedder_config(self):
        configurator = EmbeddingConfigurator()
        self.embedder_config = configurator.configure_embedder(
            self.embedder_config)

    def _initialize_app(self):
        import chromadb
        from chromadb.config import Settings
        self._set_embedder_config()
        chroma_client = chromadb.PersistentClient(
            path=self.storage_file_name,
            settings=Settings(allow_reset=self.allow_reset),
        )

        self.app = chroma_client

        try:
            self.collection = self.app.get_collection(
                name=self.type, embedding_function=self.embedder_config
            )
        except Exception:
            self.collection = self.app.create_collection(
                name=self.type, embedding_function=self.embedder_config
            )

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()
        try:
            self._add_embedding_to_collection(value, metadata)
        except Exception as e:
            logging.error(f"Error during {self.type} save: {str(e)}")

    def _add_embedding_to_collection(self, text: str, metadata: Dict[str, Any]) -> None:
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()

        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[str(uuid.uuid4())],
        )

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        if not hasattr(self, "app"):
            self._initialize_app()

        try:
            with suppress_logging():
                response = self.collection.query(
                    query_texts=query, n_results=limit)

            results = []
            for i in range(len(response["ids"][0])):
                result = {
                    "id": response["ids"][0][i],
                    "metadata": response["metadatas"][0][i],
                    "context": response["documents"][0][i],
                    "score": response["distances"][0][i],
                }
                if result["score"] >= score_threshold:
                    results.append(result)
            return results
        except Exception as e:
            logging.error(f"Error during {self.type} search: {str(e)}")
            return []

    def reset(self) -> None:
        try:
            if self.app:
                self.app.reset()
                shutil.rmtree(self.storage_file_name)
                self.app = None
                self.collection = None
        except Exception as e:
            if "attempt to write a readonly database" in str(e):
                # Ignore this specific error
                pass
            else:
                raise Exception(
                    f"An error occurred while resetting the {
                        self.type} memory: {e}"
                )

    def _create_default_embedding_function(self):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
