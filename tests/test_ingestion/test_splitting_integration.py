"""Integration tests for the chunking module.

These tests hit real external services (Google Gemini).

Requirements:
    - Network access
    - GEMINI_API_KEY environment variable set

Run with:
    uv run pytest tests/test_ingestion/ -m integration -v
To skip them:
    uv run pytest tests/test_ingestion/ -m "not integration" -v
"""

import os

import pytest

from repo_sage.ingestion.chunking import chunk_document

pytestmark = pytest.mark.integration


class TestChunkDocumentIntegration:
    """Integration tests for chunk_document against real Gemini API."""

    @pytest.fixture
    def sample_document(self) -> str:
        """Sample document for chunking tests."""
        return """
Artificial Intelligence (AI) is a branch of computer science that aims to create
machines capable of intelligent behaviour. It encompasses various subfields,
including machine learning (ML), natural language processing (NLP), and robotics.

Machine Learning is a subset of AI that focuses on developing algorithms that
allow computers to learn from and make predictions based on data. Common ML
techniques include supervised learning, unsupervised learning, and reinforcement
learning.

Natural Language Processing enables machines to understand and interpret human
language, facilitating better human-computer interactions. Applications include
chatbots, translation services, and sentiment analysis.
        """

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY environment variable not set",
    )
    def test_chunks_document_with_real_llm(self, sample_document: str) -> None:
        """Chunks a document using the real Gemini API."""
        result = chunk_document(sample_document)

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)
        assert all(len(chunk) > 0 for chunk in result)

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY environment variable not set",
    )
    def test_chunks_preserve_content(self, sample_document: str) -> None:
        """Chunks should contain text from the original document."""
        result = chunk_document(sample_document)

        # At least some key terms should appear in the chunks
        all_chunks_text = " ".join(result)
        assert "AI" in all_chunks_text or "Artificial Intelligence" in all_chunks_text
