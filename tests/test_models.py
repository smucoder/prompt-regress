import pytest
from prompt_regress.models import OpenAIProvider, AnthropicProvider, LocalProvider

@pytest.fixture
def openai_provider():
    return OpenAIProvider(model="gpt-3.5-turbo")

@pytest.fixture
def anthropic_provider():
    return AnthropicProvider()

@pytest.fixture
def local_provider():
    return LocalProvider(host="http://localhost:11434")

@pytest.mark.asyncio
async def test_openai_provider(openai_provider):
    response = await  openai_provider.agenerate("Hello, world!", model="gpt-3.5-turbo")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_anthropic_provider(anthropic_provider):
    response = await anthropic_provider.agenerate("Hello, world!", model="claude-2", max_tokens=100)
    assert isinstance(response, str)
    assert len(response) > 0 

@pytest.mark.asyncio
async def test_local_provider(local_provider):
    response = await local_provider.agenerate("Hello, world!", model="deepseek-r1:1.5b")
    assert isinstance(response.text, str)
    assert len(response.text) > 0