import os
import yaml

with open("environment.yml") as file_handle:
    environment_data = yaml.safe_load(file_handle)

for dependency in environment_data["dependencies"]:
    if isinstance(dependency, dict):
        for lib in dependency['pip']:
            os.system(f"pip install {lib}")  # pip -> pip3
            # os.system(f"pip3 install {lib}")  # pip -> pip3
