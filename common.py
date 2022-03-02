import json
import logging
import os


def try_create_directory(directory_path):
    if not os.path.exists(directory_path):
        logging.info(f"Creating directory: {directory_path}")
        os.mkdir(directory_path)


def load_json_config(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data