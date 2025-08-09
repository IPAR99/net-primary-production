from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """Load YAML configuration into a Python dict.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration as nested dict.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg