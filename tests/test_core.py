import pytest
from pathlib import Path
from prompt_regress.core import PromptRegress

@pytest.fixture
def sample_config(tmp_path):
    config = {
        "models": [
            {"name": "openai", "provider": "openai"},
            {"name": "anthropic", "provider": "anthropic"},
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

def test_get_provider_found(sample_config):
    pr = PromptRegress(sample_config)
    pr.load_config()
    provider = pr._get_provider("openai")
    assert provider["provider"] == "openai"

def test_get_provider_not_found(sample_config):
    pr = PromptRegress(sample_config)
    pr.load_config()
    with pytest.raises(ValueError):
        pr._get_provider("nonexistent")

def test_run_test_case_openai(monkeypatch, sample_config):
    pr = PromptRegress(sample_config)
    pr.load_config()
    test_case = {"inputs": [{"input": "hello"}], "name": "test", "prompt_template": "{input}"}
    model_config = {"provider": "openai", "name": "openai"}

    # Mock OpenAIProvider
    class DummyProvider:
        def generate(self, prompt, **kwargs):
            return "response"
    monkeypatch.setattr("prompt_regress.core.OpenAIProvider", DummyProvider)

    result = pr.run_test_case(test_case, model_config)
    assert isinstance(result, list)