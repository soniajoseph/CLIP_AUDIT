import os
import re
from pathlib import Path

def extract_sd(filename):
    pattern = r'^(?:\d{4}_)?neuron_\d+_layer_\d+_activation_([-]?\d+\.\d+)_([-]?\d+\.\d+)_[-]?\d+\.\d+_[-]?\d+\.\d+\.png$'
    match = re.match(pattern, filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None

def sort_and_rename_files_by_sd(neuron_dir):
    files = []
    for filename in os.listdir(neuron_dir):
        if filename.endswith('.png'):
            sd_values = extract_sd(filename)
            if sd_values:
                path = Path(neuron_dir) / filename
                files.append((path, sd_values))
    
    # Sort files by the first SD value, then by the second SD value
    sorted_files = sorted(files, key=lambda x: (x[1][0], x[1][1]))
    
    # Rename files with new numerical prefix
    for index, (file_path, (sd1, sd2)) in enumerate(sorted_files, start=1):
        old_name = file_path.name
        new_name = f"{index:04d}_" + re.sub(r'^\d{4}_', '', old_name)
        new_path = file_path.parent / new_name
        file_path.rename(new_path)
        print(f"Renamed in {neuron_dir.name}: {old_name} -> {new_name}")

def process_layer_folder(layer_dir):
    for neuron_dir in layer_dir.iterdir():
        if neuron_dir.is_dir() and neuron_dir.name.startswith("neuron_"):
            print(f"Processing {neuron_dir.name}")
            sort_and_rename_files_by_sd(neuron_dir)

def process_tinyclip_folder(tinyclip_dir):
    tinyclip_path = Path(tinyclip_dir)
    if not tinyclip_path.is_dir():
        print(f"Error: {tinyclip_dir} is not a valid directory")
        return

    for layer_dir in tinyclip_path.iterdir():
        if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
            print(f"Processing {layer_dir.name}")
            process_layer_folder(layer_dir)
            

# Usage
tinyclip_directory = f'/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/TinyCLIP40M'
process_tinyclip_folder(tinyclip_directory)