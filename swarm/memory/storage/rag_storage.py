import chromadb
from chromadb.config import Settings
import json
import datetime
import uuid
from typing import List, Dict, Any, Optional
from swarm.memory.storage.base_storage import BaseStorage
from swarm.util import PATHS


class RAGStorage(BaseStorage):
    def __init__(self, collection_name: str, chroma_client=None):
        if chroma_client is None:
            self.client = chromadb.PersistentClient(
                path=PATHS.SHORT_TERM_STORAGE,
                settings=Settings(
                    allow_reset=True
                ))
        else:
            self.client = chroma_client

        # Get or create collection
        collection_name = self._validate_collection_name(collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _generate_id(self, key: str) -> str:
        """Generate a unique ID for ChromaDB based on the key"""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def save(self, key: str, value: Any, metadata: Dict = None) -> None:
        # Convert value to string if it's not already
        if not isinstance(value, str):
            if isinstance(value, dict):
                text_value = json.dumps(value)
            else:
                text_value = str(value)
        else:
            text_value = value

        # Prepare metadata
        if metadata is None:
            metadata = {}

        metadata["key"] = key
        metadata["timestamp"] = datetime.datetime.now().isoformat()

        # Generate a unique ID based on the key
        doc_id = self._generate_id(key)

        # Add or update the document
        self.collection.upsert(
            ids=[doc_id],
            documents=[text_value],
            metadatas=[metadata]
        )


    def search(
        self,
        text: Optional[str] = None,
        key: Optional[str] = None,
        metadata: Optional[Dict] = None,
        limit: Optional[int] = 5
    ) -> List[Dict]:
        """
        Search documents by text similarity, key, or metadata.
        
        Args:
            text: Optional text to search for using vector similarity
            key: Optional key for exact match lookup
            metadata: Optional metadata filter criteria
            limit: Maximum number of results for text search (default: 5)
            
        Returns:
            List of matching documents with their metadata
        """
        if text is not None:
            # Vector similarity search
            results = self.collection.query(
                query_texts=[text],
                n_results=limit,
                where=metadata
            )
            return [
                {
                    'id': results['ids'][0][i],
                    'documents': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                for i in range(len(results['ids'][0]))
            ]
        
        elif key is not None:
            # Direct key lookup
            results = self.collection.get(where={"key": key})
        else:
            # Metadata-based lookup or empty query
            if metadata is None:
                return []
            results = self.collection.get(where=metadata)
        
        # Format results for key and metadata searches
        return [
            {
                'id': results['ids'][i],
                'documents': results['documents'][i],
                'metadata': results['metadatas'][i]
            }
            for i in range(len(results['ids']))
        ]
    
    def update(self, key: str, value: Any, metadata: Dict = None) -> None:
        # For ChromaDB, use upsert which handles both insert and update
        self.save(key, value, metadata)

    def delete(self, key: str) -> None:
        doc_id = self._generate_id(key)
        self.collection.delete(ids=[doc_id])

    def reset(self) -> None:
        self.collection.delete(where={})

    def _validate_collection_name(self, collection_name: str) -> str:
        '''
            (1) contains 3-63 characters, # Pass
            (2) starts and ends with an alphanumeric character, # Pass 
            (3) otherwise contains only alphanumeric characters, underscores or hyphens (-), 
            (4) contains no two consecutive periods (..) and 
            (5) is not a valid IPv4 address
        '''

        collection_name = collection_name.strip().lower()
        # use regex to replace all non-alphanumeric or space characters with a underscore
        import re
        collection_name = re.sub(r'[^a-z0-9]+', '_', collection_name)
        # remove all consecutive underscores
        collection_name = re.sub(r'_+', '_', collection_name)
        # remove all trailing underscores
        collection_name = collection_name.strip('_')
        # ensure the name is not a valid IPv4 address
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', collection_name):
            raise ValueError('Collection name cannot be a valid IPv4 address')
        # ensure the name is not empty
        if not collection_name:
            raise ValueError('Collection name cannot be empty')
        return collection_name

        
        