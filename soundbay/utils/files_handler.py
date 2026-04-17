import json
import yaml
import urllib

def load_config(filepath: str):
    assert filepath.endswith(("json", "yaml", "yml")), "Only json and yaml files are supported."

    is_url = filepath.startswith("http")
    is_json = filepath.endswith("json")
    is_yaml = filepath.endswith(("yaml", "yml"))

    if is_url and is_json:
        return load_json_from_url(filepath)
    elif is_url and is_yaml:
        return load_yaml_from_url(filepath)
    elif is_json:
        return load_json(filepath)
    elif is_yaml:
        return load_yaml(filepath)
    else:
        raise ValueError("File format not supported.")


def load_json_from_url(url: str):
    with urllib.request.urlopen(url) as url:
        return json.load(url)

def load_yaml_from_url(url: str):
    with urllib.request.urlopen(url) as url:
        return yaml.safe_load(url)

def load_json(filepath: str):
    with open(filepath, "r") as file:
        return json.load(file)

def load_yaml(filepath: str):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


