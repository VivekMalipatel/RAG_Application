from pathlib import Path
import yaml

def _load_prompt(key: str = None, filename: str = "prompt.yaml", base_dir: str = None) -> str:
    if base_dir is None:
        base_dir = Path(__file__).parent
    prompt_path = Path(base_dir) / filename
    with open(prompt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        prompt_data = yaml.safe_load(content)
        if key and isinstance(prompt_data, dict) and key in prompt_data:
            return prompt_data[key]
        elif isinstance(prompt_data, str):
            return prompt_data
        return ""