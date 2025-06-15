__version__ = "0.1.0"

import click
from pathlib import Path
from .core import PromptRegress

@click.group()
@click.version_option(__version__)
def cli():
    """Prompt Regress CLI - A command line interface for managing and running prompt regressions."""
    pass

@cli.command()
@click.option('--config', default='prompt-regress.yml', help='Path to the configuration file.')
def init(config):
    """Initialize the Prompt Regress configuration."""
    config_path = Path(config)
    if config_path.exists():
        click.echo(f"⚠️  Configuration already exists at {config}")
        return
    
    PromptRegress(config_path)
    click.echo(f"✅  Created configuration at {config}")
    click.echo("🔧 Edit the configuration file to add your models and test cases.")


@cli.command()
@click.option('--baseline', required=True, help='Baseline model name.')
@click.option('--target', required=True, help='Target model.')
@click.option('--config', default='prompt-regress.yml', help='Path to the configuration file.')
@click.option('--format', default='console', type=click.Choice(['console', 'json']), help='Output format')
@click.option('--fail-on-regression', is_flag=True, help='Exit with non-zero code if regressions found')
def check(baseline, target, config, format, fail_on_regression):
    "Compare outputs between two models and check for regressions."
    try:
        regress = PromptRegress(Path(config))
        results = regress.compare_models(baseline, target)
        report = regress.generate_report(results, format)

        click.echo(report)

        if fail_on_regression:
            click.echo("❌ Regressions found! Exiting with non-zero code.")
            exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    cli()