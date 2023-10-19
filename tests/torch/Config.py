import pathlib
import yaml

def read(path: pathlib.Path) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save(config: dict, path: pathlib.Path) -> None:
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)