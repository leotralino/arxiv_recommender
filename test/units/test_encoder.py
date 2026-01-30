import numpy as np
import pytest


def test_encode_single_string(vanilla_encoder):
    text = "Machine Learning in Healthcare"
    embedding = vanilla_encoder.encode(text)

    # Check that it returns a numpy array of the right dimension (MiniLM is 384)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 384)


def test_cosine_sim_identity(vanilla_encoder):
    # A vector should have a similarity of ~1.0 with itself
    vec = np.random.rand(1, 384)
    sim = vanilla_encoder.cosine_sim(vec, vec)
    assert sim == pytest.approx(1.0)


def test_cosine_sim_orthogonal(vanilla_encoder):
    # Two orthogonal vectors should have a similarity of ~0.0
    vec1 = np.array([[1, 0, 0]])
    vec2 = np.array([[0, 1, 0]])
    sim = vanilla_encoder.cosine_sim(vec1, vec2)
    assert sim == pytest.approx(0.0)


def test_get_top_k_dimensions(vanilla_encoder):
    query = np.random.rand(1, 384)
    content = np.random.rand(10, 384)
    indices = vanilla_encoder.get_top_k_similar(query, content, k=3)

    assert len(indices) == 3
    assert all(isinstance(i, (int, np.integer)) for i in indices)
