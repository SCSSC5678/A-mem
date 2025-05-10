import unittest
# import warnings # No longer needed
# warnings.simplefilter('always', DeprecationWarning) # No longer needed
from ams import AgenticMemorySystem, MemoryNote # Reverted to direct import from ams
from datetime import datetime
import json
from qdrant_client import QdrantClient # Added import
import time # Added import
import pytest # Added import
import linecache # Added import
import os # Added import

# Pytest fixture to manage linecache for the problematic module
@pytest.fixture(autouse=True, scope='session')
def manage_line_cache_for_qdrant_retriever(request):
    module_path = os.path.join(
        os.path.dirname(__file__), # Gets directory of current test file
        '..', # Go up one level to A-mem/
        'mem_sys_pkg', 
        'vector_stores', 
        'qdrant_retriever.py'
    )
    module_path = os.path.normpath(module_path) # Normalize path

    if os.path.exists(module_path):
        # print(f"\nAttempting to manage linecache for: {module_path}")
        linecache.clearcache()
        # print(f"  linecache.clearcache() called.")
        # Force a re-read of the specific line to confirm
        fresh_line = linecache.getline(module_path, 174) # Line number from warning
        # print(f"  Fresh line 174 from '{module_path}' after cache management: '{fresh_line.strip()}'")
    # else:
        # print(f"\n  Module path not found for linecache management: {module_path}")
    yield

class TestAgenticMemorySystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        # Ensure a clean Qdrant collection before initializing AgenticMemorySystem
        temp_client = QdrantClient(host="localhost", port=6333)
        collection_name = "memories" # Same as used by AgenticMemorySystem's retriever

        try:
            temp_client.delete_collection(collection_name=collection_name)
            # print(f"Attempted deletion of '{collection_name}' in setUp.")
            
            # Wait for the collection to be fully deleted
            max_wait_time = 5  # seconds
            wait_interval = 0.1 # seconds
            elapsed_time = 0
            while elapsed_time < max_wait_time:
                try:
                    temp_client.get_collection(collection_name=collection_name)
                    # If get_collection succeeds, it means the collection still exists
                    # print(f"Collection '{collection_name}' still exists, waiting...")
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval
                except Exception as e_get:
                    # Expecting an error if collection is truly deleted (e.g., "not found")
                    if "not found" in str(e_get).lower() or \
                       (hasattr(e_get, 'status_code') and e_get.status_code == 404) or \
                       (hasattr(e_get, 'code') and hasattr(e_get.code, 'value') and e_get.code().value == 5):
                        # print(f"Collection '{collection_name}' confirmed deleted.")
                        break 
                    else:
                        # Unexpected error during get_collection check
                        # print(f"Warning: Unexpected error checking deletion status for '{collection_name}': {e_get}")
                        time.sleep(wait_interval) # Still wait and retry
                        elapsed_time += wait_interval
            else:
                # Timeout reached
                print(f"Warning: Timeout waiting for collection '{collection_name}' to be deleted.")

        except Exception as e_del:
            # Ignore if initial delete_collection fails (e.g., collection didn't exist)
            # print(f"Note: Could not delete '{collection_name}' collection in setUp (may not exist or other error): {e_del}")
            pass
        
        # Now initialize the memory system. Its retriever will call _ensure_collection_exists.
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai",
            llm_model="gpt-4o-mini"
            # QdrantRetriever inside AgenticMemorySystem will use its own connection settings
        )
        # The explicit call to self.memory_system.retriever._ensure_collection_exists() is no longer needed here,
        # as it's handled during QdrantRetriever's initialization.
        
    def test_create_memory(self):
        """Test creating a new memory with complete metadata."""
        content = "Test memory content"
        tags = ["test", "memory"]
        keywords = ["test", "content"]
        links = ["link1", "link2"]
        context = "Test context"
        category = "Test category"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        
        memory_id = self.memory_system.add_note(
            content=content,
            tags=tags,
            keywords=keywords,
            links=links,
            context=context,
            category=category,
            timestamp=timestamp
        )
        
        # Verify memory was created
        self.assertIsNotNone(memory_id)
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, content)
        self.assertEqual(memory.tags, tags)
        self.assertEqual(memory.keywords, keywords)
        self.assertEqual(memory.links, links)
        self.assertEqual(memory.context, context)
        self.assertEqual(memory.category, category)
        self.assertEqual(memory.timestamp, timestamp)
        
    def test_memory_metadata_persistence(self):
        """Test that memory metadata persists through ChromaDB storage and retrieval."""
        # Create a memory with complex metadata
        content = "Complex test memory"
        tags = ["test", "complex", "metadata"]
        keywords = ["test", "complex", "keywords"]
        links = ["link1", "link2", "link3"]
        context = "Complex test context"
        category = "Complex test category"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        evolution_history = ["evolution1", "evolution2"]
        
        memory_id = self.memory_system.add_note(
            content=content,
            tags=tags,
            keywords=keywords,
            links=links,
            context=context,
            category=category,
            timestamp=timestamp,
            evolution_history=evolution_history
        )
        
        # Search for the memory using ChromaDB
        results = self.memory_system.search_agentic(content, k=1)
        self.assertGreater(len(results), 0)
        
        # Verify metadata in search results
        result = results[0]
        self.assertEqual(result['content'], content)
        self.assertEqual(result['tags'], tags)
        self.assertEqual(result['keywords'], keywords)
        self.assertEqual(result['context'], context)
        self.assertEqual(result['category'], category)
        
    def test_memory_update(self):
        """Test updating memory metadata through ChromaDB."""
        # Create initial memory
        content = "Initial content"
        memory_id = self.memory_system.add_note(content=content)
        
        # Update memory with new metadata
        new_content = "Updated content"
        new_tags = ["updated", "tags"]
        new_keywords = ["updated", "keywords"]
        new_context = "Updated context"
        
        success = self.memory_system.update(
            memory_id,
            content=new_content,
            tags=new_tags,
            keywords=new_keywords,
            context=new_context
        )
        
        self.assertTrue(success)
        
        # Verify updates in ChromaDB
        results = self.memory_system.search_agentic(new_content, k=1)
        self.assertGreater(len(results), 0)
        result = results[0]
        self.assertEqual(result['content'], new_content)
        self.assertEqual(result['tags'], new_tags)
        self.assertEqual(result['keywords'], new_keywords)
        self.assertEqual(result['context'], new_context)
        
    def test_memory_relationships(self):
        """Test memory relationships and linked memories."""
        # Create related memories
        content1 = "First memory"
        content2 = "Second memory"
        content3 = "Third memory"
        
        id1 = self.memory_system.add_note(content1)
        id2 = self.memory_system.add_note(content2)
        id3 = self.memory_system.add_note(content3)
        
        # Add relationships
        memory1 = self.memory_system.read(id1)
        memory2 = self.memory_system.read(id2)
        memory3 = self.memory_system.read(id3)
        
        memory1.links.append(id2)
        memory2.links.append(id1)
        memory2.links.append(id3)
        memory3.links.append(id2)
        
        # Update memories with relationships
        self.memory_system.update(id1, links=memory1.links)
        self.memory_system.update(id2, links=memory2.links)
        self.memory_system.update(id3, links=memory3.links)
        
        # Test relationship retrieval
        results = self.memory_system.search_agentic(content1, k=3)
        self.assertGreater(len(results), 0)
        
        # Verify relationships are maintained
        memory1_updated = self.memory_system.read(id1)
        self.assertIn(id2, memory1_updated.links)
        
    def test_memory_evolution(self):
        """Test memory evolution system with ChromaDB."""
        # Create related memories
        contents = [
            "Deep learning neural networks",
            "Neural network architectures",
            "Training deep neural networks"
        ]
        
        memory_ids = []
        for content in contents:
            memory_id = self.memory_system.add_note(content)
            memory_ids.append(memory_id)
            
        # Verify that memories have been properly evolved
        for memory_id in memory_ids:
            memory = self.memory_system.read(memory_id)
            self.assertIsNotNone(memory.tags)
            self.assertIsNotNone(memory.context)
            self.assertIsNotNone(memory.keywords)
            
        # Test evolution through search
        results = self.memory_system.search_agentic("neural networks", k=3)
        self.assertGreater(len(results), 0)
        
        # Verify evolution metadata
        for result in results:
            self.assertIsNotNone(result['tags'])
            self.assertIsNotNone(result['context'])
            self.assertIsNotNone(result['keywords'])
            
    def test_memory_deletion(self):
        """Test memory deletion from ChromaDB."""
        # Create and delete a memory
        content = "Memory to delete"
        memory_id = self.memory_system.add_note(content)
        
        # Verify memory exists
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        
        # Delete memory
        success = self.memory_system.delete(memory_id)
        self.assertTrue(success)
        
        # Verify deletion
        memory = self.memory_system.read(memory_id)
        self.assertIsNone(memory)
        
        # Verify memory is removed from ChromaDB
        results = self.memory_system.search_agentic(content, k=1)
        self.assertEqual(len(results), 0)
        
    def test_memory_consolidation(self):
        """Test memory consolidation with ChromaDB."""
        # Create multiple memories
        contents = [
            "Memory 1",
            "Memory 2",
            "Memory 3"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Force consolidation
        self.memory_system.consolidate_memories()
        
        # Verify memories are still accessible
        for content in contents:
            results = self.memory_system.search_agentic(content, k=1)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]['content'], content)
            
    def test_find_related_memories(self):
        """Test finding related memories."""
        # Create test memories
        contents = [
            "Python programming language",
            "Python data science",
            "Machine learning with Python",
            "Web development with JavaScript"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Test finding related memories
        results = self.memory_system.find_related_memories("Python", k=2)
        self.assertGreater(len(results), 0)
        
    def test_find_related_memories_raw(self):
        """Test finding related memories with raw format."""
        # Create test memories
        contents = [
            "Python programming language",
            "Python data science",
            "Machine learning with Python"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Test finding related memories in raw format
        results = self.memory_system.find_related_memories_raw("Python", k=2)
        self.assertIsNotNone(results)
        
    def test_process_memory(self):
        """Test memory processing and evolution."""
        # Create a test memory
        content = "Test memory for processing"
        memory_id = self.memory_system.add_note(content)
        
        # Get the memory
        memory = self.memory_system.read(memory_id)
        
        # Process the memory
        should_evolve, processed_memory = self.memory_system.process_memory(memory)
        
        # Verify processing results
        self.assertIsInstance(should_evolve, bool)
        self.assertIsInstance(processed_memory, MemoryNote)
        self.assertIsNotNone(processed_memory.tags)
        self.assertIsNotNone(processed_memory.context)
        self.assertIsNotNone(processed_memory.keywords)

if __name__ == '__main__':
    unittest.main()
