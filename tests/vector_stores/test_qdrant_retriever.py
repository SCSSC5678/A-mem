"""
Unit tests for QdrantRetriever.
"""
import pytest
from unittest.mock import patch, MagicMock, ANY
import uuid

# Adjust the import path based on A-Mem's structure and how it's run
# Assuming A-mem is in PYTHONPATH or tests are run from A-mem root
from memory_system.vector_stores.qdrant_retriever import QdrantRetriever
from qdrant_client import models # For creating mock ScoredPoint and Filter

# --- Constants for Tests ---
TEST_QDRANT_HOST = "localhost"
TEST_COLLECTION_NAME = "test_amem_collection"
TEST_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # A common model
TEST_EMBEDDING_DIMENSION = 384 # Dimension for all-MiniLM-L6-v2

# --- Fixtures ---

@pytest.fixture
def mock_qdrant_client_instance():
    """Mocks the QdrantClient instance."""
    mock_client = MagicMock()
    mock_client.ping.return_value = None # Simulate successful ping
    # Mock get_collection to initially raise an exception (collection not found)
    # then succeed after creation. This needs to be more sophisticated if testing
    # collection already existing vs. not. For now, assume it's created.
    mock_client.get_collection.side_effect = [Exception("Collection not found initially"), MagicMock()]
    mock_client.recreate_collection.return_value = None
    mock_client.upsert_points.return_value = None
    mock_client.search.return_value = [] # Default empty search result
    return mock_client

@pytest.fixture
def mock_sentence_transformer_instance():
    """Mocks the SentenceTransformer instance."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * TEST_EMBEDDING_DIMENSION # Dummy vector
    mock_model.get_sentence_embedding_dimension.return_value = TEST_EMBEDDING_DIMENSION
    return mock_model

@pytest.fixture
@patch('memory_system.vector_stores.qdrant_retriever.SentenceTransformer')
@patch('memory_system.vector_stores.qdrant_retriever.QdrantClient')
def retriever_instance(MockQdrantClient, MockSentenceTransformer, mock_qdrant_client_instance, mock_sentence_transformer_instance):
    """Provides a QdrantRetriever instance with mocked dependencies."""
    MockQdrantClient.return_value = mock_qdrant_client_instance
    MockSentenceTransformer.return_value = mock_sentence_transformer_instance
    
    retriever = QdrantRetriever(
        qdrant_host=TEST_QDRANT_HOST,
        collection_name=TEST_COLLECTION_NAME,
        embedding_model_name=TEST_EMBEDDING_MODEL
    )
    return retriever

# --- Test Cases ---

def test_qdrant_retriever_initialization(retriever_instance, mock_qdrant_client_instance, mock_sentence_transformer_instance):
    """Test successful initialization of QdrantRetriever."""
    # Assert QdrantClient was called
    mock_qdrant_client_instance.ping.assert_called_once()
    
    # Assert SentenceTransformer was called
    mock_sentence_transformer_instance.get_sentence_embedding_dimension.assert_called_once()
    
    # Assert _ensure_collection_exists logic (recreate_collection was called)
    mock_qdrant_client_instance.recreate_collection.assert_called_once_with(
        collection_name=TEST_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=TEST_EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE
        ),
        on_disk=True
    )
    assert retriever_instance.client is not None
    assert retriever_instance.embedding_model is not None
    assert retriever_instance.collection_name == TEST_COLLECTION_NAME
    assert retriever_instance.embedding_dimension == TEST_EMBEDDING_DIMENSION

@patch('memory_system.vector_stores.qdrant_retriever.QdrantClient')
def test_initialization_qdrant_connection_error(MockQdrantClient):
    """Test retriever initialization fails if Qdrant connection fails."""
    MockQdrantClient.return_value.ping.side_effect = Exception("Connection error")
    with pytest.raises(Exception, match="Error connecting to Qdrant"):
        QdrantRetriever(
            qdrant_host=TEST_QDRANT_HOST,
            collection_name=TEST_COLLECTION_NAME
        )

@patch('memory_system.vector_stores.qdrant_retriever.SentenceTransformer')
@patch('memory_system.vector_stores.qdrant_retriever.QdrantClient') # Mock QdrantClient to avoid its init issues
def test_initialization_embedding_model_error(MockQdrantClient, MockSentenceTransformer):
    """Test retriever initialization fails if embedding model loading fails."""
    MockQdrantClient.return_value.ping.return_value = None # Successful ping
    MockQdrantClient.return_value.get_collection.side_effect = [Exception("Collection not found initially"), MagicMock()]

    MockSentenceTransformer.side_effect = Exception("Model loading error")
    with pytest.raises(Exception, match="Error loading embedding model"):
        QdrantRetriever(
            qdrant_host=TEST_QDRANT_HOST,
            collection_name=TEST_COLLECTION_NAME
        )

def test_add_note(retriever_instance, mock_qdrant_client_instance, mock_sentence_transformer_instance):
    """Test the add_note method."""
    note_id = str(uuid.uuid4())
    text_content = "This is a test note."
    payload = {"key": "value", "raw_content": text_content}
    dummy_vector = [0.1] * TEST_EMBEDDING_DIMENSION
    mock_sentence_transformer_instance.encode.return_value.tolist.return_value = dummy_vector

    retriever_instance.add_note(note_id, text_content, payload)

    mock_sentence_transformer_instance.encode.assert_called_once_with(text_content)
    
    # Check that upsert_points was called with a list containing one PointStruct
    # We need to use ANY for the points argument if we don't want to construct an exact PointStruct for assertion
    # Or, capture the argument and inspect it.
    args, kwargs = mock_qdrant_client_instance.upsert_points.call_args
    assert kwargs['collection_name'] == TEST_COLLECTION_NAME
    assert len(kwargs['points']) == 1
    point_arg = kwargs['points'][0]
    assert point_arg.id == note_id
    assert point_arg.vector == dummy_vector
    assert point_arg.payload == payload
    assert kwargs['wait'] is True


def test_search_agentic(retriever_instance, mock_qdrant_client_instance, mock_sentence_transformer_instance):
    """Test the search_agentic method."""
    query_text = "Search query"
    limit = 5
    dummy_query_vector = [0.2] * TEST_EMBEDDING_DIMENSION
    mock_sentence_transformer_instance.encode.return_value.tolist.return_value = dummy_query_vector

    mock_scored_point = models.ScoredPoint(id=str(uuid.uuid4()), version=1, score=0.9, payload={"data": "result"}, vector=None)
    mock_qdrant_client_instance.search.return_value = [mock_scored_point]

    results = retriever_instance.search_agentic(query_text, limit)

    mock_sentence_transformer_instance.encode.assert_called_once_with(query_text)
    mock_qdrant_client_instance.search.assert_called_once_with(
        collection_name=TEST_COLLECTION_NAME,
        query_vector=dummy_query_vector,
        query_filter=None,
        limit=limit,
        with_payload=True,
        with_vector=False
    )
    
    assert len(results) == 1
    assert results[0]["id"] == mock_scored_point.id
    assert results[0]["score"] == mock_scored_point.score
    assert results[0]["payload"] == mock_scored_point.payload

def test_search_agentic_with_filters(retriever_instance, mock_qdrant_client_instance):
    """Test search_agentic with payload filters."""
    query_text = "Search with filter"
    limit = 3
    
    test_filter = models.Filter(
        must=[
            models.FieldCondition(key="tags", match=models.MatchValue(value="test"))
        ]
    )
    
    retriever_instance.search_agentic(query_text, limit, filters=test_filter)

    # We only care about the filter argument here for this specific test
    args, kwargs = mock_qdrant_client_instance.search.call_args
    assert kwargs['query_filter'] == test_filter


# TODO: Add test cases for _ensure_collection_exists when collection already exists
# TODO: Add test cases for error handling within add_note and search_agentic (e.g., Qdrant client exceptions)
