import pytest
from prompt_regress.core import PromptRegress

# Mock OpenAIProvider
class DummyProvider:
    def __init__(self, **kwargs):
        pass

    async def agenerate(self, prompt, **kwargs):
        return "response"

@pytest.fixture
def sample_config(tmp_path):
    config = {
        "models": [
            {"name": "openai", "provider": "openai"},
            {"name": "dummy", "provider": "dummy"},
        ]
    }
    config_path = tmp_path / "config.yml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

def test_load_config(sample_config):
    pr = PromptRegress(sample_config)
    pr.load_config()
    assert hasattr(pr, "config")
    assert "models" in pr.config

def test_get_provider_found(monkeypatch, sample_config):
    pr = PromptRegress(sample_config)
    config = pr.load_config()

    monkeypatch.setattr("prompt_regress.core.OpenAIProvider", DummyProvider)

    provider = pr._get_provider(config['models'][0])
    assert isinstance(provider, DummyProvider)

def test_get_provider_not_found(sample_config):
    pr = PromptRegress(sample_config)
    config = pr.load_config()
    with pytest.raises(ValueError):
        pr._get_provider(config['models'][1])

@pytest.mark.asyncio
async def test_arun_test_case_openai(monkeypatch, sample_config):
    pr = PromptRegress(sample_config)
    pr.load_config()
    test_case = {"inputs": [{"input": "hello"}], "name": "test", "prompt_template": "{input}"}
    model_config = {"provider": "openai", "name": "openai"}

    monkeypatch.setattr("prompt_regress.core.OpenAIProvider", DummyProvider)

    result = await pr.arun_test_case(test_case, model_config)
    assert isinstance(result, list)