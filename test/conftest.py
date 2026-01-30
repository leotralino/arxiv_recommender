import pytest
from arxivrec.encoder import TextEncoder


@pytest.fixture
def vanilla_encoder():
    """Fixture to reuse the encoder across tests without re-initializing."""
    return TextEncoder()  # default uses "sentence-transformers/all-MiniLM-L6-v2"
