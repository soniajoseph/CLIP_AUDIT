# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import os

# def plot_neuron_histogram_adaptive(activation_values, neuron, layer, layer_name, output_dir, num_bins):
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Calculate histogram
#     counts, bins, _ = ax.hist(activation_values, bins=num_bins, density=True, alpha=0.7)
    
#     # Calculate statistics
#     mean = np.mean(activation_values)
#     std = np.std(activation_values)
#     median = np.median(activation_values)
#     min_val = np.min(activation_values)
#     max_val = np.max(activation_values)
    
#     # # Set x-axis range to cover 99.9% of the data
#     # lower_bound = np.percentile(activation_values, 0.05)
#     # upper_bound = np.percentile(activation_values, 99.95)
#     # ax.set_xlim(lower_bound, upper_bound)
    
#     ax.set_title(f"Histogram of Activations for Neuron {neuron} in Layer {layer} ({layer_name})")
#     ax.set_xlabel("Activation Value")
#     ax.set_ylabel("Density")
    
#     stats_text = f"Mean: {mean:.2f}\nStd: {std:.2f}\nMedian: {median:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}"
#     plt.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
#              verticalalignment='top', horizontalalignment='right',
#              bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
#     # Add text to show the full range
#     full_range_text = f"Full range: [{min_val:.2f}, {max_val:.2f}]"
#     plt.text(0.05, 0.95, full_range_text, transform=ax.transAxes,
#              verticalalignment='top', horizontalalignment='left',
#              bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
#     plt.tight_layout()
    
#     # Save plot
#     filename = f"layer_{layer}_neuron_{neuron}_adaptive_scale"
#     plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
#     plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches='tight')
#     plt.close()

# def plot_multiple_neuron_histograms(h5_file_path, layer_index, neuron_indices, num_bins=100, chunk_size=1000, output_dir='plots'):
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     with h5py.File(h5_file_path, 'r') as f:
#         layers = list(f.keys())
#         if layer_index >= len(layers):
#             raise ValueError(f"Layer index {layer_index} is out of range. File contains {len(layers)} layers.")
        
#         layer_name = layers[layer_index]
#         dataset = f[layer_name]
        
#         # Process each neuron
#         for neuron in tqdm(neuron_indices, desc="Running neurons..."):
#             # Read all activation values for this neuron
#             activation_values = dataset[:, :, neuron].flatten()

#             # print length
#             # print("Length of activation values", len(activation_values))
            
#             # Plot histogram with adaptive scale
#             plot_neuron_histogram_adaptive(activation_values, neuron, layer_index, layer_name, output_dir, num_bins)
        
#         print(f"Plots saved in {output_dir} directory.")
# # Usage
# file_path = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/val/blocks.{layer}.mlp.hook_post.h5'

# layer_index = 0  # Change this to analyze different layers (0-11 for 12 layers)
# neuron_indices = [0, 1, 2, 3, 4]  # Analyze these neurons
# output_dir = 'histograms'  # Directory to save the plots
# stats = plot_multiple_neuron_histograms(file_path, layer_index, neuron_indices, num_bins=400, chunk_size=1000, output_dir=output_dir)

# # Print detailed statistics
# for neuron, neuron_stats in stats.items():
#     print(f"Neuron {neuron}:")
#     print(f"Mean: {neuron_stats['mean']:.4f}")
#     print(f"Std: {neuron_stats['std']:.4f}")
#     print(f"Min: {neuron_stats['min']:.4f}")
#     print(f"Max: {neuron_stats['max']:.4f}")
#     print(f"Total activations: {neuron_stats['count']}")
#     print("---")

# print(f"Plots saved in {output_dir} directory.")

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

def plot_neuron_histogram_adaptive(activation_values, neuron, layer, layer_name, output_dir, num_bins):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts, bins, _ = ax.hist(activation_values, bins=num_bins, density=True, alpha=0.7)
    
    mean = np.mean(activation_values)
    std = np.std(activation_values)
    median = np.median(activation_values)
    min_val = np.min(activation_values)
    max_val = np.max(activation_values)
    
    ax.set_title(f"Histogram of Activations for Neuron {neuron} in Layer {layer} ({layer_name})")
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Density")
    
    stats_text = f"Mean: {mean:.2f}\nStd: {std:.2f}\nMedian: {median:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}"
    plt.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    full_range_text = f"Full range: [{min_val:.2f}, {max_val:.2f}]"
    plt.text(0.05, 0.95, full_range_text, transform=ax.transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    
    filename = f"layer_{layer}_neuron_{neuron}_adaptive_scale"
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches='tight')
    plt.close()

    return {
        'mean': mean,
        'std': std,
        'median': median,
        'min': min_val,
        'max': max_val,
        'count': len(activation_values)
    }

def plot_neurons_histograms(h5_file_path, layers_to_process, neurons_to_process, num_bins=100, chunk_size=1000, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_file_path, 'r') as f:
        layers = list(f.keys())
        
        for layer_index in layers_to_process:
            if layer_index >= len(layers):
                print(f"Warning: Layer index {layer_index} is out of range. Skipping.")
                continue
            
            layer_name = layers[layer_index]
            print(f"Processing layer {layer_index}: {layer_name}")
            dataset = f[layer_name]
            
            num_neurons = dataset.shape[2]
            if neurons_to_process is None:
                neurons_to_process = range(num_neurons)
            else:
                neurons_to_process = [n for n in neurons_to_process if n < num_neurons]
            
            layer_stats = {}
            
            for neuron in tqdm(neurons_to_process, desc=f"Processing neurons in layer {layer_index}"):
                activation_values = dataset[:, :, neuron].flatten()
                neuron_stats = plot_neuron_histogram_adaptive(activation_values, neuron, layer_index, layer_name, output_dir, num_bins)
                layer_stats[neuron] = neuron_stats
            
            np.save(os.path.join(output_dir, f"layer_{layer_index}_stats.npy"), layer_stats)
        
        print(f"Plots and statistics saved in {output_dir} directory.")

def main():
    parser = argparse.ArgumentParser(description="Plot histograms for neuron activations in H5 file")
    parser.add_argument("file_path", type=str, help="Path to the H5 file")
    parser.add_argument("--layer", type=int, help="Specific layer to process (0-based index)")
    parser.add_argument("--all_layers", action="store_true", help="Process all layers")
    parser.add_argument("--neurons", type=int, nargs='+', help="Specific neuron indices to process")
    parser.add_argument("--all_neurons", action="store_true", help="Process all neurons in the specified layer(s)")
    parser.add_argument("--num_bins", type=int, default=400, help="Number of bins for histograms")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--output_dir", type=str, default="histograms", help="Output directory for plots and stats")

    args = parser.parse_args()

    if not args.layer and not args.all_layers:
        parser.error("Either --layer or --all_layers must be specified")

    if args.layer and args.all_layers:
        parser.error("Cannot specify both --layer and --all_layers")

    if not args.neurons and not args.all_neurons:
        parser.error("Either --neurons or --all_neurons must be specified")

    if args.neurons and args.all_neurons:
        parser.error("Cannot specify both --neurons and --all_neurons")

    with h5py.File(args.file_path, 'r') as f:
        num_layers = len(f.keys())

    if args.all_layers:
        layers_to_process = range(num_layers)
    else:
        if args.layer >= num_layers:
            parser.error(f"Layer index {args.layer} is out of range. File contains {num_layers} layers.")
        layers_to_process = [args.layer]

    neurons_to_process = args.neurons if not args.all_neurons else None

    plot_neurons_histograms(args.file_path, layers_to_process, neurons_to_process, args.num_bins, args.chunk_size, args.output_dir)

if __name__ == "__main__":
    main()