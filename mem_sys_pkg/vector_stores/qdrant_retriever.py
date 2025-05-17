"""
QdrantRetriever for A-Mem.
This class provides an interface to Qdrant for storing and retrieving A-Mem notes.
"""

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid # For generating note IDs if not provided or for Qdrant point IDs if needed
from typing import List, Dict, Any, Optional, Union
import time # Added import

class QdrantRetriever:
    """
    A retriever class that uses Qdrant as the vector store backend for A-Mem.
    """
    def __init__(self, 
                 qdrant_host: str, 
                 collection_name: str,
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 qdrant_port: int = 6333, 
                 qdrant_api_key: Optional[str] = None,
                 prefer_grpc: bool = True,
                 on_disk_payload: bool = True):
        """
        Initializes the QdrantRetriever.

        Args:
            qdrant_host (str): Hostname or IP address of the Qdrant instance.
            collection_name (str): Name of the Qdrant collection to use.
            embedding_model_name (str, optional): Name of the sentence-transformer model. 
                                                  Defaults to 'all-MiniLM-L6-v2'.
            qdrant_port (int, optional): Port for Qdrant. Defaults to 6333 (gRPC).
            qdrant_api_key (Optional[str], optional): API key for Qdrant if secured. Defaults to None.
            prefer_grpc (bool, optional): Whether to prefer gRPC for Qdrant client. Defaults to True.
            on_disk_payload (bool, optional): Whether to store payload on disk in Qdrant. Defaults to True.
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.prefer_grpc = prefer_grpc
        self.on_disk_payload = on_disk_payload

        try:
            self.client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                api_key=self.qdrant_api_key,
                prefer_grpc=self.prefer_grpc
            )
            self.client.get_collections() # Verify connection by attempting to list collections
        except Exception as e:
            # Log error or raise a custom exception for connection failure
            print(f"Error connecting to Qdrant: {e}")
            raise

        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            # Log error or raise a custom exception for model loading failure
            print(f"Error loading embedding model '{self.embedding_model_name}': {e}")
            raise

        self._ensure_collection_exists()

    def _ensure_collection_exists(self, m_value: Optional[int] = None, ef_construct_value: Optional[int] = None):
        """
        Ensures that the Qdrant collection exists and is configured correctly.
        It creates the collection only if it does not already exist.
        Optionally accepts HNSW configuration parameters for collection creation.

        Args:
            m_value (Optional[int]): HNSW M parameter (number of connections).
            ef_construct_value (Optional[int]): HNSW ef_construct parameter (construction beam size).
        """
        hnsw_config_diff = None
        if m_value is not None and ef_construct_value is not None:
            hnsw_config_diff = models.HnswConfigDiff(m=m_value, ef_construct=ef_construct_value)
        elif m_value is not None or ef_construct_value is not None:
            print(f"Warning: Only one of m_value ({m_value}) or ef_construct_value ({ef_construct_value}) was provided. "
                  f"Both are needed to customize HNSW config. Proceeding without custom HNSW config.")

        try:
            self.client.get_collection(collection_name=self.collection_name)
            # If collection exists, this method currently does not modify it, even if HNSW params are different.
            # The benchmarking script should handle deletion and recreation if specific HNSW params are required for a run.
            # print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            # Check if the exception indicates the collection was not found
            if "not found" in str(e).lower() or \
               (hasattr(e, 'status_code') and e.status_code == 404) or \
               (hasattr(e, 'code') and hasattr(e.code, 'value') and e.code().value == 5): # gRPC StatusCode.NOT_FOUND

                # print(f"Collection '{self.collection_name}' does not exist. Attempting to create.")
                try:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=self.embedding_dimension,
                            distance=models.Distance.COSINE
                        ),
                        on_disk_payload=self.on_disk_payload,
                        hnsw_config=hnsw_config_diff # Pass HNSW config here
                    )
                    # print(f"Collection '{self.collection_name}' created successfully with HNSW: {hnsw_config_diff}.")
                except Exception as creation_e:
                    if "collection data already exists" in str(creation_e).lower():
                        # print(f"Attempting to re-create '{self.collection_name}' due to existing data on disk.")
                        try:
                            self.client.delete_collection(collection_name=self.collection_name)
                            time.sleep(0.5) 
                            self.client.create_collection(
                                collection_name=self.collection_name,
                                vectors_config=models.VectorParams(
                                    size=self.embedding_dimension,
                                    distance=models.Distance.COSINE
                                ),
                                on_disk_payload=self.on_disk_payload,
                                hnsw_config=hnsw_config_diff # And here
                            )
                            # print(f"Collection '{self.collection_name}' re-created successfully with HNSW: {hnsw_config_diff} after clearing existing data.")
                        except Exception as recreate_e:
                            print(f"Error re-creating collection '{self.collection_name}': {recreate_e}")
                            raise recreate_e
                    else:
                        print(f"Error creating collection '{self.collection_name}' after it was deemed non-existent: {creation_e}")
                        raise creation_e
            else:
                # An unexpected error occurred while trying to get collection info
                print(f"Unexpected error when checking existence of collection '{self.collection_name}': {e}")
                raise
    
    def add_document(self, document: str, metadata: dict, doc_id: Optional[str] = None, vector: Optional[List[float]] = None): # Matched AgenticMemorySystem's call
        """
        Adds a document (embedding and payload) to the Qdrant collection.
        This method name matches what AgenticMemorySystem.add_note calls on its retriever.
        The `doc_id` parameter corresponds to `note_id` in A-Mem.

        Args:
            document (str): The textual content of the note to be embedded.
            metadata (Dict[str, Any]): Metadata associated with the A-Mem note.
                                      This should include all A-Mem note attributes such as
                                      'content', 'semantic_context', 'tags', 'keywords',
                                      'creation_timestamp', 'last_access_timestamp',
                                      'retrieval_count', and importantly, 'related_memory_ids'
                                      (a list of strings) if inter-note links are to be stored.
            doc_id (Optional[str]): Unique identifier for the note.
            vector (Optional[List[float]]): Optional pre-computed embedding vector. If provided,
                                            the document will not be embedded by this method.
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        try:
            if vector is None:
                vector = self.embedding_model.encode(document).tolist()
            # else: use provided vector

            point = models.PointStruct(
                id=doc_id,
                vector=vector,
                payload=metadata # metadata from AgenticMemorySystem is the payload
            )
            print(f"[QDRANT_ADD] ID: {doc_id}, Content Snippet: {document[:100]}...") # ADDED LOGGING

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            # print(f"Document '{doc_id}' added/updated in collection '{self.collection_name}'.")
        except Exception as e:
            # Log error or raise a custom exception
            print(f"Error adding document '{doc_id}' to Qdrant: {e}")
            raise

    def search(self, query_text: str, k: int = 5, filters: Optional[models.Filter] = None) -> Dict[str, List[List[Any]]]:
        """
        Performs a semantic search in the Qdrant collection.
        This method is named 'search' to match what AgenticMemorySystem calls.
        The return format is adapted to be compatible with what AgenticMemorySystem expects
        from the ChromaRetriever (a dictionary with nested lists).

        Args:
            query_text (str): The text to search for.
            k (int): The maximum number of results to return. (Corresponds to 'limit' in Qdrant)
            filters (Optional[models.Filter], optional): Qdrant filter conditions. Defaults to None.

        Returns:
            Dict[str, List[List[Any]]]]: A dictionary structured like ChromaDB's output:
                {
                    "ids": [List[str_ids]],
                    "metadatas": [List[payload_dicts]],
                    "distances": [List[score_floats]] 
                }
                Note: 'distances' here will contain Qdrant's similarity scores (higher is better).
        """
        try:
            query_vector = self.embedding_model.encode(query_text).tolist()
            print(f"[QDRANT_SEARCH] Query Text: {query_text[:100]}...") # ADDED LOGGING
            
            search_params = models.SearchParams(hnsw_ef=128, exact=True) # Explicitly set search-time HNSW ef, force exact

            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=filters,
                limit=k,
                with_payload=True,
                search_params=search_params
            )
            
            actual_points_list = []
            if hasattr(search_results, 'points') and search_results.points is not None:
                actual_points_list = search_results.points
            elif hasattr(search_results, 'hits') and search_results.hits is not None: 
                actual_points_list = search_results.hits
            
            print(f"[QDRANT_SEARCH_RAW_RESULTS] For query '{query_text[:50]}...': {actual_points_list[:5]}") # ADDED LOGGING (log first 5 raw results)

            results_ids: List[str] = []
            results_payloads: List[Dict[str, Any]] = []
            results_scores: List[float] = []

            if actual_points_list: 
                first_item = actual_points_list[0]
                if hasattr(first_item, 'id') and hasattr(first_item, 'score'): 
                    for point_data in actual_points_list:
                        if hasattr(point_data, 'id') and hasattr(point_data, 'score'): 
                            results_ids.append(str(point_data.id))
                            results_payloads.append(point_data.payload if point_data.payload is not None else {})
                            results_scores.append(point_data.score)
                        else:
                            print(f"Warning: Mixed item types in search results. Expected ScoredPoint-like, got {type(point_data)}: {point_data}")
                elif isinstance(first_item, tuple): 
                    for point_tuple in actual_points_list:
                        if isinstance(point_tuple, tuple) and len(point_tuple) >= 2: 
                            point_id = point_tuple[0]
                            point_score = point_tuple[1]
                            point_payload = point_tuple[2] if len(point_tuple) > 2 and point_tuple[2] is not None else {}
                            
                            results_ids.append(str(point_id))
                            results_payloads.append(point_payload)
                            results_scores.append(point_score)
                        else:
                            print(f"Warning: Malformed or unexpected tuple in search results: {point_tuple}")
                else:
                    print(f"Warning: Unexpected item type in search results list. First item type: {type(first_item)}")
            
            return {
                "ids": [results_ids],
                "metadatas": [results_payloads],
                "distances": [results_scores]
            }
        except Exception as e:
            print(f"Error searching in Qdrant: {e}")
            return {"ids": [[]], "metadatas": [[]], "distances": [[]]}

    def delete_document(self, doc_id: str): # Matched AgenticMemorySystem's call
        """
        Deletes a document from the Qdrant collection by its ID.
        Named to match what AgenticMemorySystem.update and delete call.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[doc_id]),
                wait=True
            )
            # print(f"Document '{doc_id}' deleted from collection '{self.collection_name}'.")
        except Exception as e:
            print(f"Error deleting document '{doc_id}' from Qdrant: {e}")

    def add_documents_batch(self, documents_data: List[Dict[str, Any]]):
        """
        Adds a batch of documents to the Qdrant collection.

        Args:
            documents_data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                                 contains 'document' (str), 'metadata' (dict),
                                                 'doc_id' (str), and optionally 'vector' (List[float])
                                                 for a note.
        """
        points_to_upsert = []
        for item in documents_data:
            doc_content = item.get('document')
            metadata = item.get('metadata')
            doc_id = item.get('doc_id')
            vector = item.get('vector') # Get optional pre-computed vector

            if not all([doc_content, metadata, doc_id]):
                print(f"Warning: Skipping item in batch due to missing data: {item}")
                continue

            try:
                if vector is None:
                    # If no vector is provided, encode the document
                    vector = self.embedding_model.encode(doc_content).tolist()
                # else: use provided vector

                point = models.PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=metadata
                )
                points_to_upsert.append(point)
            except Exception as e:
                print(f"Error processing document '{doc_id}' for batch add: {e}")
                continue

        if not points_to_upsert:
            # print("No valid points to upsert in the batch.")
            return

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points_to_upsert,
                wait=True
            )
            # print(f"Successfully upserted {len(points_to_upsert)} documents in a batch.")
        except Exception as e:
            print(f"Error upserting batch of documents to Qdrant: {e}")
            raise


if __name__ == '__main__':
    # Example Usage (requires Qdrant instance running)
    # This is for basic testing and would be removed or moved to a test/example script.
    try:
        retriever = QdrantRetriever(
            qdrant_host="localhost", # Or your Qdrant host
            collection_name="amem_test_collection_main_search" # Use a distinct name for testing
        )
        print(f"QdrantRetriever initialized. Embedding dimension: {retriever.embedding_dimension}")

        # Example: Add a note
        sample_note_id = str(uuid.uuid4())
        sample_text = "This is a test note about Qdrant and A-Mem integration."
        sample_payload = {
            "a_mem_note_id": sample_note_id, 
            "content": sample_text, 
            "semantic_context": "A test context.",
            "tags": ["test", "qdrant", "a-mem"],
            "keywords": ["qdrant", "retriever"],
            "creation_timestamp": "2025-05-09T12:00:00Z",
            "last_access_timestamp": "2025-05-09T12:00:00Z",
            "retrieval_count": 0,
            "related_memory_ids": []
        }
        retriever.add_document(document=sample_text, metadata=sample_payload, doc_id=sample_note_id)
        print(f"Added note: {sample_note_id}")

        # Example: Add another note
        sample_note_id_2 = str(uuid.uuid4())
        sample_text_2 = "Qdrant is a vector database used for similarity search."
        sample_payload_2 = {
            "a_mem_note_id": sample_note_id_2,
            "content": sample_text_2,
            "semantic_context": "Information about Qdrant.",
            "tags": ["qdrant", "vector-database"],
            "keywords": ["qdrant", "database"],
            "creation_timestamp": "2025-05-09T12:01:00Z",
            "last_access_timestamp": "2025-05-09T12:01:00Z",
            "retrieval_count": 0,
            "related_memory_ids": [sample_note_id] 
        }
        retriever.add_document(document=sample_text_2, metadata=sample_payload_2, doc_id=sample_note_id_2)
        print(f"Added note: {sample_note_id_2}")


        # Example: Search for notes using the new 'search' method
        query = "Tell me about Qdrant"
        results = retriever.search(query_text=query, k=5) 
        print(f"\nSearch results for '{query}':")
        
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                res_id = results["ids"][0][i]
                res_meta = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                res_score = results["distances"][0][i] if results["distances"] and results["distances"][0] else float('nan')
                
                print(f"  ID: {res_id}, Score: {res_score:.4f}, Payload Content: {res_meta.get('content', 'N/A')[:50]}...")
        else:
            print("No results found or unexpected format.")

    except Exception as e:
        print(f"An error occurred during example usage: {e}")
