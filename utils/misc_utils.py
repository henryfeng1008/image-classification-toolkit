import os
import yaml


def load_config(config_path, check=False):
    f = open(config_path)
    config = yaml.load(f, Loader=yaml.FullLoader)
    if check:
        print(config)
    return config