import yaml
from pathlib import Path
from typing import Dict

def load_config(config_path: str = None) -> Dict:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)