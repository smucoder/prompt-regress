import pytest
from prompt_regress.models import OpenAIProvider, AnthropicProvider, LocalProvider

@pytest.fixture
def openai_provider():
    return OpenAIProvider()

@pytest.fixture
def anthropic_provider():
    return AnthropicProvider()

@pytest.fixture
def local_provider():
    return LocalProvider()


def test_openai_provider(openai_provider):
    response = openai_provider.generate("Hello, world!", model_name="gpt-3.5-turbo")
    assert isinstance(response, str)
    assert len(response) > 0

def test_anthropic_provider(anthropic_provider):
    response = anthropic_provider.generate("Hello, world!", model_name="claude-2")
    assert isinstance(response, str)
    assert len(response) > 0 

def test_local_provider(local_provider):
    response = local_provider.generate("Hello, world!", model_name="deepseek-r1:1.5b")
    assert isinstance(response, str)
    assert len(response) > 0