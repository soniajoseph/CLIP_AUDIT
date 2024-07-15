import os
import argparse
from pathlib import Path
import math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_neuron_figure(neuron_dir):
    # Get all PNG files in the neuron directory
    output_file = os.path.join(neuron_dir, f"{Path(neuron_dir).name}_compiled.png")
    # if output file exists, break
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists")
        return

    png_files = sorted([f for f in os.listdir(neuron_dir) if f.endswith('.png')])
    
    if not png_files:
        print(f"No PNG files found in {neuron_dir}")
        return
    
    # Determine grid size
    n_images = len(png_files)
    cols = math.ceil(math.sqrt(n_images))
    rows = math.ceil(n_images / cols)
    
    # Create a new figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    fig.suptitle(f"Neuron {Path(neuron_dir).name}", fontsize=16)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten() if n_images > 1 else [axes]
    
    # Add images to the plot
    for i, png_file in enumerate(png_files):
        img = Image.open(os.path.join(neuron_dir, png_file))
        axes[i].imshow(np.array(img))
        axes[i].axis('off')
        axes[i].set_title(png_file, fontsize=8)
    
    # Remove any unused subplots
    for i in range(n_images, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Compiled figure saved as {output_file}")

def process_layer(tinyclip_dir, layer_num):
    layer_dir = Path(tinyclip_dir) / f"layer_{layer_num}"
    if not layer_dir.is_dir():
        print(f"Error: Layer directory {layer_dir} not found")
        return

    print(f"Processing {layer_dir.name}")
    for neuron_dir in layer_dir.iterdir():
        if neuron_dir.is_dir() and neuron_dir.name.startswith("neuron_"):
            print(f"Processing {neuron_dir.name}")
            create_neuron_figure(neuron_dir)

def main():
    parser = argparse.ArgumentParser(description="Compile neuron images for a specific layer in TinyCLIP.")
    parser.add_argument("tinyclip_directory", type=str, help="Path to the TinyCLIP directory")
    parser.add_argument("--layer_num", type=int, required=True, help="Layer number to process")

    args = parser.parse_args()

    tinyclip_directory = args.tinyclip_directory
    layer_num = args.layer_num

    if not Path(tinyclip_directory).is_dir():
        print(f"Error: {tinyclip_directory} is not a valid directory")
        return

    process_layer(tinyclip_directory, layer_num)

if __name__ == "__main__":
    main()