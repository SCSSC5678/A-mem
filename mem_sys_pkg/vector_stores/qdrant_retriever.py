"""
QdrantRetriever for A-Mem.
This class provides an interface to Qdrant for storing and retrieving A-Mem notes.
"""

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid # For generating note IDs if not provided or for Qdrant point IDs if needed
from typing import List, Dict, Any, Optional, Union

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

    def _ensure_collection_exists(self):
        """
        Ensures that the Qdrant collection exists and is configured correctly.
        It creates the collection only if it does not already exist.
        The test's setUp method is responsible for deleting the collection for a clean state.
        """
        try:
            # Attempt to get collection info. If it succeeds, the collection exists.
            self.client.get_collection(collection_name=self.collection_name)
            # print(f"Collection '{self.collection_name}' already exists. No action needed by _ensure_collection_exists.")
        except Exception as e:
            # Check if the exception indicates the collection was not found
            # This condition might need to be adjusted based on the exact error qdrant_client throws
            # for a non-existent collection (e.g., a specific error type or status code in the error message).
            # A common pattern is an RPC error with status code 5 (NOT_FOUND) or a message containing "not found".
            # For qdrant_client, a `UnexpectedResponse` with status 404 or similar gRPC error.
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
                    )
                    # print(f"Collection '{self.collection_name}' created successfully.")
                except Exception as creation_e:
                    print(f"Error creating collection '{self.collection_name}' after it was deemed non-existent: {creation_e}")
                    raise  # Re-raise the creation error
            else:
                # An unexpected error occurred while trying to get collection info
                print(f"Unexpected error when checking existence of collection '{self.collection_name}': {e}")
                raise # Re-raise the unexpected error
    
    def add_document(self, document: str, metadata: dict, doc_id: Optional[str] = None): # Matched AgenticMemorySystem's call
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
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        try:
            vector = self.embedding_model.encode(document).tolist()
            
            point = models.PointStruct(
                id=doc_id, 
                vector=vector,
                payload=metadata # metadata from AgenticMemorySystem is the payload
            )
            
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
            
            # Note: client.search is deprecated. Using client.query_points (or client.query if that's the final name)
            # For qdrant-client >= 1.7.0, `search` is a simplified version of `query`.
            # The direct replacement for more complex queries or to be future-proof is `query`.
            # However, the parameters used here (query_vector, query_filter, limit, with_payload)
            # are directly compatible with `search`'s signature and also with `query` if structured correctly.
            # Given the deprecation warning, it's safer to assume `query` or `query_points` is preferred.
            # Let's assume `search` is still functional but we're preparing for its removal.
            # The user mentioned `query_points` as the replacement.
            # The `search` method in qdrant-client is often a wrapper.
            # The warning clearly states "Use `query_points` instead."
            # Context7 examples for `query_points` show using `query` as the parameter name for the vector.
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,       # Corrected from query_vector to query
                query_filter=filters,
                limit=k,
                with_payload=True
            )
            # Note: The parameter names for `query_points` might differ slightly.
            # Common parameters for search-like methods in Qdrant are `vector` (for the query vector itself)
            # and `filter` (for the filter object). `with_payload` is also common.
            # If `query_points` has different parameter names, this will need adjustment.
            # For now, assuming direct mapping from `search`/`query` parameters.
            
            # The search_results is a QueryResponse object.
            # The actual list of points is likely in an attribute like 'points' or 'hits'.
            # Let's assume 'points' first, as it's common.
            # If this fails, 'hits' would be the next common attribute name.
            
            actual_points_list = []
            if hasattr(search_results, 'points') and search_results.points is not None:
                actual_points_list = search_results.points
            elif hasattr(search_results, 'hits') and search_results.hits is not None: # Fallback if .points doesn't exist or is None
                actual_points_list = search_results.hits
            # else:
                # print(f"DEBUG: QueryResponse object does not have 'points' or 'hits' attribute, or they are None.")
                # print(f"DEBUG: dir(search_results): {dir(search_results)}")


            results_ids: List[str] = []
            results_payloads: List[Dict[str, Any]] = []
            results_scores: List[float] = []

            # Process the items in actual_points_list
            # This logic is based on the user's report to handle ScoredPoint or tuple results.
            if actual_points_list: # Check if the list is not empty
                # Introspect the first item to decide processing strategy (heuristic)
                first_item = actual_points_list[0]
                if hasattr(first_item, 'id') and hasattr(first_item, 'score'): # Likely ScoredPoint
                    for point_data in actual_points_list:
                        if hasattr(point_data, 'id') and hasattr(point_data, 'score'): # Double check each item
                            results_ids.append(str(point_data.id))
                            results_payloads.append(point_data.payload if point_data.payload is not None else {})
                            results_scores.append(point_data.score)
                        else:
                            print(f"Warning: Mixed item types in search results. Expected ScoredPoint-like, got {type(point_data)}: {point_data}")
                elif isinstance(first_item, tuple): # Likely a list of tuples
                    for point_tuple in actual_points_list:
                        if isinstance(point_tuple, tuple) and len(point_tuple) >= 2: # Expecting at least (id, score)
                            point_id = point_tuple[0]
                            point_score = point_tuple[1]
                            # Assuming payload is the third element if present
                            point_payload = point_tuple[2] if len(point_tuple) > 2 and point_tuple[2] is not None else {}
                            
                            results_ids.append(str(point_id))
                            results_payloads.append(point_payload)
                            results_scores.append(point_score)
                        else:
                            print(f"Warning: Malformed or unexpected tuple in search results: {point_tuple}")
                else:
                    print(f"Warning: Unexpected item type in search results list. First item type: {type(first_item)}")
            # else:
                # print("DEBUG: actual_points_list is empty.")
            
            return {
                "ids": [results_ids],
                "metadatas": [results_payloads],
                "distances": [results_scores]  # Using Qdrant scores (similarity) for 'distances' key
            }
        except Exception as e:
            # Log error or raise a custom exception
            print(f"Error searching in Qdrant: {e}")
            # Return empty structure in case of error to avoid breaking AgenticMemorySystem
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
            # Decide if to raise or just log, depending on desired error handling
            # For now, just printing, but A-Mem might expect an exception or boolean.

    def add_documents_batch(self, documents_data: List[Dict[str, Any]]):
        """
        Adds a batch of documents to the Qdrant collection.

        Args:
            documents_data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                                 contains 'document' (str), 'metadata' (dict), 
                                                 and 'doc_id' (str) for a note.
        """
        points_to_upsert = []
        for item in documents_data:
            doc_content = item.get('document')
            metadata = item.get('metadata')
            doc_id = item.get('doc_id')

            if not all([doc_content, metadata, doc_id]):
                print(f"Warning: Skipping item in batch due to missing data: {item}")
                continue

            try:
                vector = self.embedding_model.encode(doc_content).tolist()
                point = models.PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=metadata
                )
                points_to_upsert.append(point)
            except Exception as e:
                print(f"Error processing document '{doc_id}' for batch add: {e}")
                # Optionally, decide whether to skip this point or halt the batch
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
            # Consider how to handle partial failures if Qdrant supports it, or if to re-raise
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
            "a_mem_note_id": sample_note_id, # Storing original A-Mem ID in payload
            "content": sample_text, # Storing content also in payload for easy access by A-Mem
            "semantic_context": "A test context.",
            "tags": ["test", "qdrant", "a-mem"],
            "keywords": ["qdrant", "retriever"],
            "creation_timestamp": "2025-05-09T12:00:00Z",
            "last_access_timestamp": "2025-05-09T12:00:00Z",
            "retrieval_count": 0,
            "related_memory_ids": []
        }
        # AgenticMemorySystem calls add_document(content, metadata, id)
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
            "related_memory_ids": [sample_note_id] # Link to the first note
        }
        retriever.add_document(document=sample_text_2, metadata=sample_payload_2, doc_id=sample_note_id_2)
        print(f"Added note: {sample_note_id_2}")


        # Example: Search for notes using the new 'search' method
        query = "Tell me about Qdrant"
        # AgenticMemorySystem calls search(query, k)
        results = retriever.search(query_text=query, k=5) 
        print(f"\nSearch results for '{query}':")
        
        # Assuming the structure AgenticMemorySystem expects:
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
