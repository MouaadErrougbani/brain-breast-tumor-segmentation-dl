import yaml

def load_config(file_path):
    with open(file=file_path, mode='r') as file: 
        config = yaml.safe_load(file)

    return config