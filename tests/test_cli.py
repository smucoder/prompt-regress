from click.testing import CliRunner
from prompt_regress.cli import cli

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "Usage" in result.output

def test_init_creates_config_file(tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "test-config.yml"
    result = runner.invoke(cli, ['init', '--config', str(config_path)])
    assert result.exit_code == 0
    assert config_path.exists()
    assert "✅ Created configuration at" in result.output

def test_init_existing_config(tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "test-config.yml"
    config_path.write_text("dummy: config")
    result = runner.invoke(cli, ['init', '--config', str(config_path)])
    assert "already exists" in result.output

def test_check_missing_models(tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "test-config.yml"
    config_path.write_text("models: []")
    result = runner.invoke(cli, [
        'check', '--baseline', 'foo', '--target', 'bar', '--config', str(config_path)
    ])
    assert result.exit_code != 0
    assert "not found in configuration" in result.output

def test_version_option():
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    print(result.output)
    print(result.exception)
    assert result.exit_code == 0
    assert "version" in result.output.lower()