import pytest
from unittest.mock import MagicMock, patch
from chat import answer_query, retrieve_context


@patch("chat.embedding_functions.SentenceTransformerEmbeddingFunction")
@patch("chat.chromadb.PersistentClient")
def test_retrieve_context(mock_chroma_client, mock_ef):
    # Instantiate the mocked client and its internal collection mechanisms
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.return_value = mock_db
    mock_db.get_collection.return_value = mock_collection

    # Simulate ChromaDB's output format: {'documents': [['chunk1', 'chunk2']]}
    mock_collection.query.return_value = {
        "documents": [
            [
                "SECTION 1: RETURN POLICY - Customers may return items within 30 days.",
                "SECTION 3: WARRANTY - All electronics come with a 1-year warranty.",
            ]
        ]
    }

    context = retrieve_context("What is the return policy?")

    assert "SECTION 1: RETURN POLICY" in context
    assert "--- NEXT POLICY CHUNK ---" in context
    mock_collection.query.assert_called_once_with(
        query_texts=["What is the return policy?"], n_results=2
    )


@patch("chat.client.chat")
@patch("chat.retrieve_context")
def test_answer_query(mock_retrieve_context, mock_chat):
    mock_retrieve_context.return_value = "Mocked policy context"
    mock_chat.return_value = {
        "message": {"content": "The return policy allows standard returns within 30 days."}
    }

    # Verify execution of the generation layer without raising errors
    answer_query("return policy query")

    mock_retrieve_context.assert_called_once_with("return policy query")
    mock_chat.assert_called_once()