
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import argparse


import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import argparse

def plot_neuron_histogram(h5_file_path, layer_num, neuron_num, output_dir, num_bins=100, density=True, log=False):
    with h5py.File(h5_file_path, 'r') as f:
        layer_name = f"blocks.{layer_num}.hook_mlp_out"
        
        if layer_name not in f:
            print(f"Layer {layer_name} not found in the H5 file.")
            return
        
        dataset = f[layer_name]
        
        if neuron_num >= dataset.shape[2]:
            print(f"Neuron {neuron_num} is out of range for layer {layer_name}.")
            return
        
        neuron_data = dataset[:, :, neuron_num]
        
        output_file = os.path.join(output_dir, f"{layer_name}_neuron_{neuron_num}_histogram_log_{log}.png")
        
        plt.figure(figsize=(12, 6))
        
        # Get a color map
        cmap = plt.get_cmap('tab20')
        
        # Plot histogram for each patch
        for i in range(neuron_data.shape[1]):
            patch_data = neuron_data[:, i]
            color = cmap(i / neuron_data.shape[1])
            sns.histplot(patch_data, bins=num_bins, kde=True, color=color, stat='density' if density else 'count', 
                         alpha=0.5, label=f'Patch {i}')
        
        if log:
            plt.yscale('log')
        
        plt.title(f"Histogram of Neuron {neuron_num} Activations in {layer_name}")
        plt.xlabel("Activation Values")
        if log:
            plt.ylabel("Log(Density)" if density else "Log(Count)")
        else:
            plt.ylabel("Density" if density else "Count")
        
        # Add mean and median lines
        mean_activation = np.mean(neuron_data)
        median_activation = np.median(neuron_data)
        plt.axvline(mean_activation, color='r', linestyle='--', label=f'Mean: {mean_activation:.2f}')
        plt.axvline(median_activation, color='g', linestyle='--', label=f'Median: {median_activation:.2f}')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Histogram for neuron {neuron_num} in {layer_name} saved as {output_file}")

# The main function remains the same

def main():
    parser = argparse.ArgumentParser(description="Plot histogram of a specific neuron's activations in a given layer")
    parser.add_argument("--file_path", type=str, required=False, default='/network/scratch/s/sonia.joseph/CLIP_AUDIT/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/val/blocks.{layer}.hook_mlp_out.h5', help="Path to the H5 file")
    parser.add_argument("--layer_num", type=int, required=False, default=8, help="Layer number")
    parser.add_argument("--neuron_num", type=int, required=False, default=1, help="Neuron number")
    parser.add_argument("--output_dir", type=str, default="neuron_histograms", help="Output directory")
    parser.add_argument("--num_bins", type=int, default=100, help="Number of bins for the histogram")
    parser.add_argument("--density", action="store_true", help="Plot density instead of count")
    parser.add_argument("--log", action="store_true", help="Use logarithmic scale for y-axis")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_neuron_histogram(args.file_path, args.layer_num, args.neuron_num, args.output_dir, 
                          args.num_bins, args.density, args.log)

if __name__ == "__main__":
    main()