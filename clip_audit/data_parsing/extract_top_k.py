'''
This is a folder to extract the top k.
'''

original_dir = ...
new_dir = ...

import os
import shutil
import re

# Define the source and destination directories
source_dir = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/TinyCLIP40M'
dest_dir = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/TinyCLIP40m-topk'

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Compile a regular expression to match the file name pattern
file_pattern = re.compile(r'neuron_\d+_layer_(\d+)_top_k_only\.png')

# Iterate through the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        match = file_pattern.match(file)
        if match:
            # Extract layer number from the file name
            layer_num = match.group(1)
            
            # Create the destination layer directory if it doesn't exist
            layer_dest_dir = os.path.join(dest_dir, f'layer_{layer_num}')
            os.makedirs(layer_dest_dir, exist_ok=True)
            
            # Copy the file to the new location
            src_file = os.path.join(root, file)
            dest_file = os.path.join(layer_dest_dir, file)
            shutil.copy2(src_file, dest_file)

print("File reorganization complete.")