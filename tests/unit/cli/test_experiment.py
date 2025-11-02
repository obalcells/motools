"""Minimal smoke tests for experiment CLI."""

import yaml
from typer.testing import CliRunner

from motools.cli.experiment import app, load_config

runner = CliRunner()


def test_load_yaml_config(tmp_path):
    """Test loading YAML configuration."""
    config_path = tmp_path / "config.yaml"
    config_data = {
        "base_config": {"prepare_dataset": {"dataset_loader": "test.loader"}},
        "param_grid": {"param1": [1, 2, 3]},
    }
    config_path.write_text(yaml.safe_dump(config_data))

    loaded = load_config(config_path)
    assert loaded == config_data


def test_template_command(tmp_path):
    """Test generating template."""
    output_path = tmp_path / "template.yaml"
    result = runner.invoke(app, ["template", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()

    config = yaml.safe_load(output_path.read_text())
    assert "base_config" in config
    assert "param_grid" in config
