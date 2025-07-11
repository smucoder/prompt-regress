import pytest
from prompt_regress.metrics import SimilarityMetrics


@pytest.fixture
def metrics():
    return SimilarityMetrics(embedding_model='Qwen/Qwen3-Embedding-0.6B')

def test_text_similarity_identical(metrics):
    assert metrics.text_similarity(["hello world"], ["hello world"])[0] == 1.0

def test_text_similarity_different(metrics):
    score = metrics.text_similarity(["hello world"], ["goodbye world"])[0]
    assert 0 <= score < 1.0

def test_text_similarity_empty(metrics):
    with pytest.raises(ValueError):
        metrics.text_similarity([], [])
        metrics.text_similarity(["a"], []) == 0.0
        metrics.text_similarity([], ["b"]) == 0.0

def test_semantic_similarity_identical(metrics):
    # This assumes semantic_similarity returns 1.0 for identical strings
    assert metrics.semantic_similarity(["test"], ["test"]) >= 0.90

def test_semantic_similarity_different(metrics):
    assert 0 <= metrics.semantic_similarity(["cat"], ["dog"]) < 1.0

def test_semantic_similarity_empty(metrics):
    with pytest.raises(ValueError):
        metrics.semantic_similarity([], []) 
        metrics.semantic_similarity(["a"], [])
        metrics.semantic_similarity([], ["b"])