import keyword
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
import json
import logging
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

from qdrant_client import models as qdrant_models # Moved import to module level
from sentence_transformers import SentenceTransformer # Moved import to module level
from qdrant_client import QdrantClient # Moved import to module level


class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """Initialize a new memory note with its associated metadata.
        
        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[Dict]): References to related memories
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format YYYYMMDDHHMM
            last_accessed (Optional[str]): Last access time in format YYYYMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        
        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

class QdrantRetriever:
    """Retriever for Qdrant vector database.
    
    This class provides methods for adding, retrieving, and deleting documents from Qdrant.
    """
    
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "memories",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 qdrant_api_key: Optional[str] = None):
        """Initialize the QdrantRetriever.
        
        Args:
            qdrant_host (str): Host for Qdrant service
            qdrant_port (int): Port for Qdrant service
            collection_name (str): Name of the Qdrant collection to use
            embedding_model_name (str): Name of the embedding model to use
            qdrant_api_key (Optional[str]): API key for Qdrant service (if secured)
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.qdrant_api_key = qdrant_api_key
        
        # Imports moved to module level

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key, prefer_grpc=False) # prefer_grpc=False for wider compatibility
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()

        # Ensure collection exists
        try:
            self.client.get_collection(collection_name=self.collection_name)
            logger.info(f"Qdrant collection '{self.collection_name}' already exists.")
        except Exception as e:
            if "not found" in str(e).lower() or "status_code=404" in str(e).lower():
                logger.info(f"Qdrant collection '{self.collection_name}' not found. Creating now...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(size=self.vector_size, distance=qdrant_models.Distance.COSINE)
                )
                logger.info(f"Qdrant collection '{self.collection_name}' created successfully with vector size {self.vector_size}.")
            else:
                logger.error(f"Error checking Qdrant collection '{self.collection_name}': {e}")
                raise
        
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text."""
        return self.embedding_model.encode(text).tolist()

    def add_document(self, document: str, metadata: Dict[str, Any], doc_id: str, vector: Optional[List[float]] = None):
        """Add a document to Qdrant.
        
        Args:
            document (str): The document text (used if vector is not provided).
            metadata (Dict[str, Any]): Payload for the document.
            doc_id (str): ID for the document.
            vector (Optional[List[float]]): Pre-computed embedding vector.
        """
        if vector is None:
            vector = self._get_embedding(document) # Embed the 'document' field if no vector given
        
        point = qdrant_models.PointStruct(
            id=doc_id,
            vector=vector,
            payload=metadata # The entire metadata dict becomes the payload
        )
        self.client.upsert(collection_name=self.collection_name, points=[point]) # Changed to upsert
        logger.debug(f"Upserted document with ID '{doc_id}' to Qdrant collection '{self.collection_name}'.")
        
    def add_documents_batch(self, documents_data: List[Dict[str, Any]]):
        """Add multiple documents to Qdrant in a batch.
        
        Args:
            documents_data (List[Dict[str, Any]]): List of document data. Each dict should have
                                                 'doc_id', 'document' (text for embedding if no vector),
                                                 'metadata' (payload), and optionally 'vector'.
        """
        points_batch = []
        for doc_data in documents_data:
            doc_id = doc_data['doc_id']
            document_text = doc_data['document']
            metadata = doc_data['metadata']
            vector = doc_data.get('vector')

            if vector is None:
                vector = self._get_embedding(document_text)
            
            points_batch.append(qdrant_models.PointStruct(
                id=doc_id,
                vector=vector,
                payload=metadata
            ))
        
        if points_batch:
            self.client.upsert(collection_name=self.collection_name, points=points_batch) # Changed to upsert
            logger.debug(f"Upserted batch of {len(points_batch)} documents to Qdrant collection '{self.collection_name}'.")
            
    def delete_document(self, doc_id: str):
        """Delete a document from Qdrant.
        
        Args:
            doc_id (str): ID of the document to delete.
        """
        self.client.delete( # Changed to delete
            collection_name=self.collection_name,
            points_selector=qdrant_models.PointIdsList(points=[doc_id])
        )
        logger.debug(f"Deleted document with ID '{doc_id}' from Qdrant collection '{self.collection_name}'.")
            
    def search(self, query_text: Optional[str] = None, query_vector: Optional[List[float]] = None, k: int = 5, filters: Optional[qdrant_models.Filter] = None) -> List[Dict[str, Any]]:
        """Search for documents in Qdrant.
        
        Args:
            query_text (Optional[str]): The query text. If provided, it will be embedded.
            query_vector (Optional[List[float]]): A pre-computed query vector.
                                                 One of query_text or query_vector must be provided.
            k (int): Number of results to return.
            filters (Optional[qdrant_models.Filter]): Qdrant filter conditions.
            
        Returns:
            List[Dict[str, Any]]: A list of search results, where each result is a dictionary
                                  containing 'id', 'score', and 'payload'.
        """
        if query_vector is None:
            if query_text is None:
                raise ValueError("Either query_text or query_vector must be provided for search.")
            query_vector = self._get_embedding(query_text)

        # Use query_points instead of search
        # query_points returns a list of ScoredPoint objects directly
        # The `query` parameter in query_points is for the vector
        scored_points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector, # Renamed from query_vector
            query_filter=filters,
            limit=k,
            with_payload=True
            # with_vector is not a direct parameter for query_points
            # By default, query_points may not return vectors unless specifically requested
            # or if using a vector selector. For our current use, not requesting vectors is fine.
        ).points # Access the .points attribute from QueryResponse
        
        # Transform ScoredPoint results into a list of dictionaries
        processed_results = []
        for scored_point in scored_points: # Iterate over the list of ScoredPoint
            processed_results.append({
                "id": scored_point.id,
                "score": scored_point.score,
                "payload": scored_point.payload
            })
        return processed_results

class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 qdrant_api_key: Optional[str] = None,
                 collection_name: str = "memories"):
        """Initialize the memory system.
        
        Args:
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
            qdrant_host: Host for Qdrant service
            qdrant_port: Port for Qdrant service
            qdrant_api_key: API key for Qdrant service (if secured)
            collection_name: Name of the Qdrant collection to use
        """
        self.memories = {}
        
        # Store Qdrant connection parameters
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
        self.embedding_model_name_ams = model_name
        self.collection_name = collection_name

        # Initialize QdrantRetriever
        self.retriever = QdrantRetriever(
            qdrant_host=self.qdrant_host,
            qdrant_port=self.qdrant_port,
            collection_name=self.collection_name,
            embedding_model_name=self.embedding_model_name_ams,
            qdrant_api_key=self.qdrant_api_key
        )
        
        # Mock LLM controller for PoC
        self.llm_controller = type('MockLLMController', (), {'llm': type('MockLLM', (), {'get_completion': lambda *args, **kwargs: json.dumps({"keywords": [], "context": "General", "tags": []})})()})()
        
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        
    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        # Mock implementation for PoC
        return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, vector: Optional[List[float]] = None, **kwargs) -> str:
        """Add a new memory note.

        Args:
            content (str): The main text content of the memory.
            time (str, optional): Creation time in format YYYYMMDDHHMM. Defaults to None.
            vector (Optional[List[float]], optional): Optional pre-computed embedding vector.
                                                      If provided, the content will not be embedded
                                                      by the retriever. Defaults to None.
            **kwargs: Additional fields for the MemoryNote.

        Returns:
            str: The ID of the newly added memory note.
        """
        # Create MemoryNote
        if time is not None:
            kwargs['timestamp'] = time
        note = MemoryNote(content=content, **kwargs)

        # Process memory for potential evolution (this might update note metadata)
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note

        # Prepare a rich payload for Qdrant, aligning with STP Table 2
        # The 'document' field for embedding is note.content
        # The payload should contain all other structured attributes.
        qdrant_payload = {
            "note_id": note.id, # Store note.id in payload for easier reference
            "content_text": note.content, # Store original content text
            "keywords": note.keywords,
            "tags": note.tags,
            "contextual_description": note.context, # Assuming MemoryNote.context is the contextual_description
            "linked_notes_ids": [link_id for link_id in note.links if isinstance(link_id, str)], # Assuming links are a list of IDs
            "creation_timestamp": note.timestamp, # Ensure Qdrant compatible format if strict
            "last_updated_timestamp": note.last_accessed, # Ensure Qdrant compatible format
            # "source_document_uri": kwargs.get("source_document_uri"), # If provided in **kwargs
            # For image_references, we need to structure it as per STP Table 2.
            # Assuming **kwargs might contain 'image_references' as List[Dict]
            # e.g., [{"image_uri": "...", "vlm_description": "..."}]
            "image_references": kwargs.get("image_references", []) 
        }
        # Add any other relevant fields from note or kwargs to qdrant_payload
        for key, value in kwargs.items():
            if key not in qdrant_payload and key not in ['timestamp']: # Avoid overwriting existing or handled fields
                qdrant_payload[key] = value
        
        # Add to Qdrant using the retriever, passing the optional vector
        # The 'document' argument to add_document is used for embedding if 'vector' is None.
        self.retriever.add_document(document=note.content, metadata=qdrant_payload, doc_id=note.id, vector=vector)

        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def add_notes_batch(self, notes_data: List[Dict[str, Any]]) -> List[str]:
        """Add a batch of memory notes.

        Args:
            notes_data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                              contains data for a memory note. Each dictionary
                                              must include 'content' (str) and can optionally
                                              include 'id' (str), 'time' (str), 'vector' (List[float]),
                                              and other MemoryNote attributes.

        Returns:
            List[str]: A list of IDs for the newly added memory notes.
        """
        notes_to_process = []
        processed_note_ids = []

        for item_data in notes_data:
            content = item_data.get('content')
            if content is None:
                logger.warning(f"Skipping item in batch due to missing 'content': {item_data}")
                continue

            # Extract arguments for MemoryNote constructor
            mn_id = item_data.get('id')
            mn_keywords = item_data.get('keywords')
            mn_links = item_data.get('links') # MemoryNote expects 'links', payload might have 'linked_notes_ids'
            mn_retrieval_count = item_data.get('retrieval_count')
            mn_timestamp = item_data.get('time') # 'time' key from ingest_into_amem_tool's item_for_batch
            mn_last_accessed = item_data.get('last_accessed')
            # 'contextual_description' from payload should map to 'context' for MemoryNote
            mn_context = item_data.get('contextual_description', item_data.get('context')) 
            mn_evolution_history = item_data.get('evolution_history')
            mn_category = item_data.get('category')
            mn_tags = item_data.get('tags')
            
            vector = item_data.get('vector')

            # Create MemoryNote object with only its defined parameters
            note = MemoryNote(
                content=content, 
                id=mn_id,
                keywords=mn_keywords,
                links=mn_links,
                retrieval_count=mn_retrieval_count,
                timestamp=mn_timestamp,
                last_accessed=mn_last_accessed,
                context=mn_context,
                evolution_history=mn_evolution_history,
                category=mn_category,
                tags=mn_tags
            )

            # Process memory for potential evolution (this might update note attributes)
            evo_label, note = self.process_memory(note)
            self.memories[note.id] = note # Add to in-memory store

            # Prepare the rich Qdrant payload. Start with all data from item_data (which includes the original payload)
            # and then ensure critical/standardized fields from the 'note' object are present.
            qdrant_payload_for_item = item_data.copy() # Start with everything from item_for_batch
            
            # Ensure essential fields from the processed 'note' object are in the payload,
            # potentially overwriting if MemoryNote processing changed them (e.g., generated ID, default timestamp).
            qdrant_payload_for_item["note_id"] = note.id # Use the (potentially generated) ID from MemoryNote
            qdrant_payload_for_item["content_text"] = note.content # Use content from MemoryNote
            qdrant_payload_for_item["keywords"] = note.keywords
            qdrant_payload_for_item["tags"] = note.tags
            qdrant_payload_for_item["contextual_description"] = note.context # Standardized field name
            qdrant_payload_for_item["creation_timestamp"] = note.timestamp
            qdrant_payload_for_item["last_updated_timestamp"] = note.last_accessed
            
            # Remove keys that are not part of the Qdrant payload itself but were used for MemoryNote creation or are redundant
            qdrant_payload_for_item.pop('content', None) # 'content' was for MemoryNote, Qdrant payload uses 'content_text'
            qdrant_payload_for_item.pop('id', None)      # 'id' was for MemoryNote, Qdrant payload uses 'note_id'
            qdrant_payload_for_item.pop('vector', None)  # 'vector' is handled separately by QdrantRetriever
            qdrant_payload_for_item.pop('time', None)    # 'time' was mapped to 'timestamp' for MemoryNote

            # 'image_references', 'source_uri', 'agent_id' etc. from the original payload in item_data will be preserved.

            notes_to_process.append({
                'document': note.content, # Text used for embedding if vector is None
                'metadata': qdrant_payload_for_item, # The rich payload
                'doc_id': note.id, # Qdrant point ID
                'vector': vector 
            })
            processed_note_ids.append(note.id)

            if evo_label:
                 self.evo_cnt += 1

        # Perform batch upsert to Qdrant
        if notes_to_process:
            self.retriever.add_documents_batch(notes_to_process)

        # Check for consolidation after the batch
        if self.evo_cnt >= self.evo_threshold:
             self.consolidate_memories()
             self.evo_cnt = 0 # Reset evolution counter after consolidation

        return processed_note_ids

    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents"""
        # Reset Qdrant collection
        self.retriever = QdrantRetriever(
            qdrant_host=self.qdrant_host,
            qdrant_port=self.qdrant_port,
            collection_name=self.collection_name,
            embedding_model_name=self.embedding_model_name_ams,
            qdrant_api_key=self.qdrant_api_key
        )
        # Re-add all memory documents with their complete metadata using batch add
        all_memories_data = []
        for memory in self.memories.values():
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": memory.evolution_history,
                "category": memory.category,
                "tags": memory.tags
            }
            all_memories_data.append({
                'document': memory.content,
                'metadata': metadata,
                'doc_id': memory.id,
                'vector': None 
            })
        if all_memories_data:
            self.retriever.add_documents_batch(all_memories_data)

    def find_related_memories(self, query: str, k: int = 5, filters: Optional[Any] = None) -> Tuple[str, List[int]]:
        """Find related memories using Qdrant retrieval.

        Args:
            query (str): The text to search for.
            k (int): The maximum number of results to return.
            filters (Optional[Any], optional): Qdrant filter conditions. Defaults to None.

        Returns:
            Tuple[str, List[int]]: A tuple containing:
                - str: A formatted string of the retrieved memory contents and metadata.
                - List[int]: A list of indices corresponding to the order of memories in the formatted string.
        """
        if not self.memories:
            return "", []

        try:
            # Get results from QdrantRetriever, passing filters
            results = self.retriever.search(query_text=query, k=k, filters=filters)

            # Convert to list of memories
            memory_str = ""
            indices = [] # Indices will correspond to the order in the formatted string

            # The new self.retriever.search returns List[Dict[str, Any]]
            # Each dict has 'id', 'score', 'payload'
            # 'payload' contains the rich metadata stored in Qdrant.
            if results:
                for i, hit in enumerate(results): # results is now a list of hits
                    payload = hit.get('payload', {})
                    # Format memory string using data from payload
                    memory_str += f"memory index:{i}\ttalk start time:{payload.get('creation_timestamp', '')}\tmemory content: {payload.get('content_text', '')}\tmemory context: {payload.get('contextual_description', '')}\tmemory keywords: {str(payload.get('keywords', []))}\tmemory tags: {str(payload.get('tags', []))}\n"
                    indices.append(i) # Store the simple index
            return memory_str, indices
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def search_agentic(self, query: str, k: int = 5, filters: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Search for memories using Qdrant retrieval.
           The 'filters' argument here is expected to be in the Qdrant `models.Filter` format
           if passed directly to `self.retriever.search`. If it's a dict, it needs translation.
           For simplicity, assuming `filters` is already a Qdrant `models.Filter` or None.
        """
        if not self.memories and not self.retriever: # Check if retriever is available
            logger.warning("No memories or retriever available for search_agentic.")
            return []
            
        try:
            # Get results from QdrantRetriever.
            # The `filters` argument for `self.retriever.search` expects a qdrant_models.Filter object.
            # If `filters` passed to `search_agentic` is a dict, it needs translation.
            # For now, assuming it's passed correctly or is None.
            qdrant_search_results = self.retriever.search(query_text=query, k=k, filters=filters)
            
            # Process Qdrant search results (List[Dict[str, Any]])
            # Each dict in qdrant_search_results has 'id', 'score', 'payload'
            processed_memories = []
            if qdrant_search_results:
                for hit in qdrant_search_results:
                    payload = hit.get('payload', {})
                    memory_dict = {
                        'id': hit.get('id'), # This is the Qdrant point ID, should match note_id
                        'content': payload.get('content_text', ''), # Main text content
                        'context': payload.get('contextual_description', ''),
                        'keywords': payload.get('keywords', []),
                        'tags': payload.get('tags', []),
                        'timestamp': payload.get('creation_timestamp', ''),
                        'category': payload.get('category', 'Uncategorized'), # If 'category' is in payload
                        'score': hit.get('score'),
                        'is_neighbor': False, # This field might be specific to how results are used later
                        'image_references': payload.get('image_references', []) # Retrieve image_references
                        # Add other fields from payload as needed by the application
                    }
                    processed_memories.append(memory_dict)
            
            return processed_memories # Already sliced to k by retriever.search
            
        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve.
        
        Args:
            note: The memory note to process
            
        Returns:
            Tuple[bool, MemoryNote]: (should_evolve, processed_note)
        """
        # For first memory or testing, just return the note without evolution
        if not self.memories:
            return False, note
            
        try:
            # Get nearest neighbors
            neighbors_text, indices = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not indices:
                return False, note
                
            # Mock evolution decision for PoC
            return False, note
                
        except Exception as e:
            # For testing purposes, catch all exceptions and return the original note
            logger.error(f"Error in process_memory: {str(e)}")
            return False, note
