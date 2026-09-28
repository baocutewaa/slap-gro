"""Configuration loading and validation for SLAP-GRO."""

from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return the root path of the project."""
    return Path(__file__).resolve().parent.parent.parent.parent


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Safely load a YAML configuration file."""
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_contract(contract_path: str = "configs/data_contract.yaml") -> Dict[str, Any]:
    """Load the data contract definition."""
    return load_yaml(contract_path)


def load_paper_config(config_path: str = "configs/paper_replication.yaml") -> Dict[str, Any]:
    """Load the paper replication mode configuration."""
    return load_yaml(config_path)


def load_real_config(config_path: str = "configs/real_data.yaml") -> Dict[str, Any]:
    """Load the real dataset mode configuration."""
    return load_yaml(config_path)


def get_config(mode: str = "real") -> Dict[str, Any]:
    """Get the configuration for the specified mode ('paper' or 'real')."""
    if mode.lower() in ("paper", "paper_replication"):
        return load_paper_config()
    elif mode.lower() in ("real", "real_data", "real_mode"):
        return load_real_config()
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'paper' or 'real'.")
