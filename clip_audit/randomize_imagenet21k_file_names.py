import os
import re
import base64
import hashlib
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json


def encode_filename(components_string, existing_hashes):
    """
    Create a unique encoded filename, avoiding collisions with existing hashes.
    """
    base_hash = hashlib.sha256(components_string.encode()).hexdigest()[:16]
    
    # If the base hash is already used, append numbers until we find a unique one
    if base_hash in existing_hashes:
        counter = 1
        while True:
            new_hash = f"{base_hash}_{counter}"
            if new_hash not in existing_hashes:
                return new_hash
            counter += 1
    
    return base_hash




def generate_random_hash(filename, existing_hashes):
    """Generate a unique hash that isn't in existing_hashes"""
    while True:
        hash_input = f"{filename}{os.urandom(16)}"
        new_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        if new_hash not in existing_hashes:
            return new_hash


def crop_title_from_image(image_path):
    """Crop out the title area from the image"""
    with Image.open(image_path) as img:
        width, height = img.size
        # Crop out approximately the top 10% where titles usually are
        title_height = int(height * 0.025)
        cropped_img = img.crop((0, title_height, width, height))
        return cropped_img

def randomize_filenames(source_dir, dest_dir, mapping_file='filename_mapping.json'):
    """
    Randomize filenames in source_dir and copy to dest_dir using an obscured encoding scheme.
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Dictionary to store mapping of encoded to original filename
    filename_mapping = {}
    
    # First, collect all valid files
    all_files = []
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if not filename.lower().endswith('.svg') and "_no_heatmap" not in filename:
                all_files.append((root, filename))
    
    used_hashes = set()
    # Process files with progress bar
    for root, filename in tqdm(all_files, desc="Processing files"):
        # Get full paths
        source_file_path = os.path.join(root, filename)
        rel_file_path = os.path.relpath(source_file_path, source_dir)
        
        # Extract components from the relative path
        path_parts = rel_file_path.split('/')
        layer_num = int(path_parts[0].replace('layer_', ''))
        neuron_num = int(path_parts[1].replace('neuron_', ''))
        hook_type = path_parts[2]
        sampling_type = path_parts[3]
        extreme_type = path_parts[4]
        
        # Create the string to encode in the correct format
        encoding_string = f"neuron_{neuron_num}_layer_{layer_num}_{extreme_type}_{sampling_type}_{hook_type}"
        encoded_name = encode_filename(encoding_string, used_hashes)
        used_hashes.add(encoded_name)
        
        # Get file extension
        file_extension = Path(filename).suffix
        
        # Create new filenames
        new_filename = f"{encoded_name}{file_extension}"
        new_filename_no_heatmap = f"{encoded_name}_no_heatmap{file_extension}"
        
        # Get destination paths (flat structure)
        dest_file_path = os.path.join(dest_dir, new_filename)
        dest_file_path_no_heatmap = os.path.join(dest_dir, new_filename_no_heatmap)
        
        # If it's an image file, crop and save
        if file_extension.lower() in ['.png', '.jpg', '.jpeg']:
            try:
                # Process the heatmap version
                cropped_img = crop_title_from_image(source_file_path)
                if cropped_img:
                    cropped_img.save(dest_file_path)
                
                # Process the no_heatmap version if it exists
                no_heatmap_source = source_file_path.replace(file_extension, f"_no_heatmap{file_extension}")
                if os.path.exists(no_heatmap_source):
                    cropped_img = crop_title_from_image(no_heatmap_source)
                    if cropped_img:
                        cropped_img.save(dest_file_path_no_heatmap)
                    
            except Exception as e:
                print(f"Error processing image {source_file_path}: {e}")
                # If there's an error, just copy the original files
                shutil.copy2(source_file_path, dest_file_path)
                if os.path.exists(no_heatmap_source):
                    shutil.copy2(no_heatmap_source, dest_file_path_no_heatmap)
        else:
            # Copy non-image files directly
            shutil.copy2(source_file_path, dest_file_path)
        
        # Store mapping
        filename_mapping[encoded_name] = {
            'original_name': filename,
            'original_path': rel_file_path,
            'new_name': new_filename,
            'new_name_no_heatmap': new_filename_no_heatmap,
            'decoded_components': {
                'layer': layer_num,
                'neuron': neuron_num,
                'extreme_type': extreme_type,
                'sampling_type': sampling_type,
                'hook_type': hook_type
            }
        }
    
    # Save mapping to JSON file
    mapping_path = os.path.join(dest_dir, mapping_file)
    with open(mapping_path, 'w') as f:
        json.dump(filename_mapping, f, indent=4)
    
    print(f"\nProcessed {len(filename_mapping)} files")
    print(f"Mapping saved to {mapping_path}")
    
    return filename_mapping



def verify_mapping(source_dir, mapping_file):
    """Verify the mapping against source files with detailed path comparison"""
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Get all mapped paths
    mapped_paths = {info['original_path']: encoded_name 
                   for encoded_name, info in mapping.items()}
    
    # Collect source files
    source_files = []
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if not filename.lower().endswith('.svg') and "_no_heatmap" not in filename:
                rel_path = os.path.relpath(os.path.join(root, filename), source_dir)
                source_files.append(rel_path)
    
    print(f"Total source files: {len(source_files)}")
    print(f"Total mapped files: {len(mapped_paths)}")
    
    # Check specifically for the files we thought were missing
    supposedly_missing = [
        "layer_1/neuron_1/hook_resid_post/max/bottom/neuron_695_layer_1_bottom_max_hook_resid_post.png",
        "layer_1/neuron_1/hook_resid_post/max/top/neuron_695_layer_1_top_max_hook_resid_post.png",
        "layer_2/neuron_3/hook_resid_post/max/bottom/neuron_122_layer_2_bottom_max_hook_resid_post.png",
        "layer_2/neuron_3/hook_resid_post/max/top/neuron_122_layer_2_top_max_hook_resid_post.png",
        "layer_2/neuron_9/hook_mlp_out/max/bottom/neuron_468_layer_2_bottom_max_hook_mlp_out.png",
        "layer_2/neuron_9/hook_mlp_out/max/top/neuron_468_layer_2_top_max_hook_mlp_out.png",
        "layer_6/neuron_3/hook_mlp_out/max/bottom/neuron_578_layer_6_bottom_max_hook_mlp_out.png",
        "layer_6/neuron_3/hook_mlp_out/max/top/neuron_578_layer_6_top_max_hook_mlp_out.png"
    ]
    
    print("\nChecking supposedly missing files:")
    for path in supposedly_missing:
        if path in mapped_paths:
            print(f"\nFile IS mapped: {path}")
            print(f"Encoded name: {mapped_paths[path]}")
        else:
            print(f"\nFile NOT mapped: {path}")
            
            # Check if filename exists anywhere in mapping
            filename = os.path.basename(path)
            for mapped_path in mapped_paths:
                if os.path.basename(mapped_path) == filename:
                    print(f"But found filename in different path: {mapped_path}")
    
    # Find any genuinely missing files
    missing = set(source_files) - set(mapped_paths.keys())
    if missing:
        print("\nGenuinely missing files:")
        for path in sorted(missing):
            print(path)
    else:
        print("\nNo genuinely missing files found!")

def get_full_filename(encoded_name):
    """
    Convert encoded filename back to original path format.
    
    Args:
        encoded_name (str): The encoded filename (with or without extension)
        
    Returns:
        str: Original format path/filename
    """
    try:
        # Remove file extension if present
        encoded_name = encoded_name.split('.')[0]
        # Remove _no_heatmap suffix if present
        encoded_name = encoded_name.replace('_no_heatmap', '')
        
        # Try loading from mapping file first
        mapping_path = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons_randomized/filename_mapping.json'
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
                if encoded_name in mapping:
                    return mapping[encoded_name]['original_path']
        
        # Direct decoding if no mapping found
        components = encoded_name.split('_')
        
        # Extract components
        neuron_num = int(components[1])
        layer_num = int(components[3])
        extreme_type = components[4]
        sampling_type = components[5]
        hook_type = components[6]
        
        # Reconstruct original path format
        original_path = (f"layer_{layer_num}/"
                        f"neuron_{neuron_num}/"
                        f"{hook_type}/"
                        f"{sampling_type}/"
                        f"{extreme_type}")
        
        return original_path
        
    except Exception as e:
        print(f"Error processing filename {encoded_name}: {e}")
        return None


# # Example usage:
source_directory = "/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons"  # Your source directory
destination_directory = "/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons_randomized"  # Where to save randomized files

# mapping = randomize_filenames(source_directory, destination_directory)

# test
file_name = '4d4628970eb41c1e.png'
mapping_file = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons_randomized/filename_mapping.json'
orig_path = get_full_filename(file_name)
print(orig_path)