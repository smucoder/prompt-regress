import pytest
from prompt_regress.metrics import SimilarityMetrics


@pytest.fixture
def metrics():
    return SimilarityMetrics()

def test_text_similarity_identical(metrics):
    assert metrics.text_similarity("hello world", "hello world") == 1.0

def test_text_similarity_different(metrics):
    score = metrics.text_similarity("hello world", "goodbye world")
    assert 0 <= score < 1.0

def test_text_similarity_empty(metrics):
    assert metrics.text_similarity("", "") == 1.0
    assert metrics.text_similarity("a", "") == 0.0
    assert metrics.text_similarity("", "b") == 0.0

def test_semantic_similarity_identical(metrics):
    # This assumes semantic_similarity returns 1.0 for identical strings
    assert metrics.semantic_similarity("test", "test") == 1.0

def test_semantic_similarity_different(metrics):
    assert 0 <= metrics.semantic_similarity("cat", "dog") < 1.0

def test_semantic_similarity_empty(metrics):
    assert metrics.semantic_similarity("", "") == 1.0
    assert 0 <= metrics.semantic_similarity("a", "") < 1.0 
    assert 0 <= metrics.semantic_similarity("", "b") < 1.0 