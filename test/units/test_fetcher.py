from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from arxivrec.dataset.fetcher import ArxivFetcher
from arxivrec.topic import Topic


@pytest.fixture
def topic():
    return Topic(categories=["cs.LG", "cs.AI"])


@pytest.fixture
def fetcher(topic):
    return ArxivFetcher(topic=topic, lookback_days=1, max_results=5)


@patch("arxivrec.dataset.fetcher.arxiv.Client")
@patch("arxivrec.dataset.fetcher.arxiv.Search")
def test_fetch_returns_dataframe(mock_search, mock_client, fetcher):
    # Mock arxiv result
    mock_result = MagicMock()
    mock_result.entry_id = "http://arxiv.org/abs/1234.5678"
    mock_result.title = "Test Title"
    mock_result.authors = [MagicMock(name="Author", name__="Alice")]
    mock_result.authors[0].name = "Alice"
    mock_result.summary = "Test abstract\nwith newline."
    mock_result.published = pd.Timestamp.now(tz="UTC")
    mock_result.primary_category = "cs.LG"
    mock_result.pdf_url = "http://arxiv.org/pdf/1234.5678"

    # Mock client.results to return a list of mock results
    mock_client.return_value.results.return_value = [mock_result]

    df = fetcher.fetch()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "id" in df.columns
    assert "title" in df.columns
    assert "authors" in df.columns
    assert "abstract" in df.columns
    assert "published" in df.columns
    assert "primary_category" in df.columns
    assert "url" in df.columns
    assert "combined_text" in df.columns
    assert df.iloc[0]["title"] == "Test Title"
    assert df.iloc[0]["authors"] == ["Alice"]


@patch("arxivrec.dataset.fetcher.arxiv.Client")
@patch("arxivrec.dataset.fetcher.arxiv.Search")
def test_fetch_breaks_on_old_paper(mock_search, mock_client, fetcher):
    # Mock arxiv result with old published date
    mock_result = MagicMock()
    mock_result.entry_id = "http://arxiv.org/abs/0000.0000"
    mock_result.title = "Old Paper"
    mock_result.authors = [MagicMock(name="Author", name__="Bob")]
    mock_result.authors[0].name = "Bob"
    mock_result.summary = "Old abstract."
    mock_result.published = pd.Timestamp("2000-01-01", tz="UTC")
    mock_result.primary_category = "cs.AI"
    mock_result.pdf_url = "http://arxiv.org/pdf/0000.0000"

    mock_client.return_value.results.return_value = [mock_result]

    df = fetcher.fetch()
    # Should break before adding old paper
    assert df.empty


@patch("arxivrec.dataset.fetcher.arxiv.Client")
@patch("arxivrec.dataset.fetcher.arxiv.Search")
def test_fetch_with_fallback(mock_search, mock_client, fetcher):
    mock_result = MagicMock()
    mock_result.entry_id = "http://arxiv.org/abs/9999.9999"
    mock_result.title = "Fallback Paper"
    mock_result.authors = [MagicMock(name="Alice")]
    mock_result.authors[0].name = "Alice"
    mock_result.summary = "Fallback abstract."
    mock_result.published = pd.Timestamp.now(tz="UTC")
    mock_result.primary_category = "cs.LG"
    mock_result.pdf_url = "http://arxiv.org/pdf/9999.9999"

    # 2. Use side_effect to simulate the sequence of calls
    # Call 1: Returns empty list (triggers fallback)
    # Call 2: Returns our mock result
    mock_client.return_value.results.side_effect = [[], [mock_result]]

    df = fetcher.fetch()

    # 4. Assertions
    assert not df.empty
    assert df.iloc[0]["title"] == "Fallback Paper"
    # Verify that the search was actually called twice
    assert mock_client.return_value.results.call_count == 2
