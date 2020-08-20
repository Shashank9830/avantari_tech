import shutil
import json
import os

with open('k_groups.json') as file:
    medoids, clusters = json.load(file)

dataset_source = 'dataset'
base_dir = 'K Groups'

for i in range(len(medoids)):

    new_group_dir = f"{base_dir}/{i+1}"
    os.mkdir(new_group_dir)

    for file in clusters[i]:
        shutil.copy(f"{dataset_source}/{file}.jpg", f"{new_group_dir}/{file}.jpg")
