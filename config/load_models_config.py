import yaml

def _load_models_config():
    """Load and validate models configuration from YAML file."""
    try:
        with open('config/models.yaml', 'r') as f:
            models_config = yaml.safe_load(f)
        return models_config
    except FileNotFoundError:
        raise FileNotFoundError("Error: models.yaml not found in config/ directory")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error: Invalid YAML format in models.yaml: {e}")