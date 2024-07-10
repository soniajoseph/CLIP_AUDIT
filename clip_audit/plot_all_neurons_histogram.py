import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import seaborn as sns

import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import os
import numpy as np

import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import os
import numpy as np

import pandas as pd

def print_h5_keys(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        print("Top-level keys:")
        for key in f.keys():
            print(f"- {key}")
            
        # If there's a 'blocks' group, print its keys too
        if 'blocks' in f:
            print("\nKeys in 'blocks' group:")
            for block_key in f['blocks'].keys():
                print(f"- {block_key}")
                # If you want to go one level deeper, uncomment the following lines
                # if isinstance(f['blocks'][block_key], h5py.Group):
                #     for sub_key in f['blocks'][block_key].keys():
                #         print(f"  - {sub_key}")

def plot_histograms(h5_file_path, output_dir, num_bins=100, density=True, log=False):
    print_h5_keys(h5_file_path)
    with h5py.File(h5_file_path, 'r') as f:
        layers = [key for key in f.keys() if key != 'image_indices']
        
        palette = sns.color_palette("husl", len(layers))
        
        print(layers)

        for layer_name in tqdm(layers, desc="Processing layers"):

            # Make log optional

            # If file exists, then skip it
            layer_output_file = os.path.join(output_dir, f"{layer_name}_histogram_log_{log}.png")
            if os.path.exists(layer_output_file):
                print(f"Skipping {layer_name} as {layer_output_file} already exists")
                continue

            dataset = f[layer_name]
            
            # Flatten the entire dataset
            layer_data = dataset[:].ravel()
            # all_data.append(layer_data)

            # length of dataset
            n = len(layer_data)
            print(f"Length of dataset: {n}")

            plt.figure(figsize=(12, 6))
            
            n, bins, patches = plt.hist(layer_data, bins=num_bins, alpha=0.7, 
                                        color=palette[layers.index(layer_name)], density=density)
            
            if log:
                plt.yscale('log')
            
            plt.title(f"Histogram of Neuron Activations for {layer_name}")
            plt.xlabel("Activation Values")
            if log:
                plt.ylabel("Log(Density)" if density else "Log(Count)")
            else:
                plt.ylabel("Density" if density else "Count")
            plt.tight_layout()
            

            plt.savefig(layer_output_file, dpi=300, bbox_inches='tight')
            # save as svg
            plt.savefig(layer_output_file.replace('.png', '.svg'), dpi=300, bbox_inches='tight')
            # save as svg
            plt.close()
            print(f"Histogram for {layer_name} saved as {layer_output_file}")
        
        # plt.figure(figsize=(20, 10))
        
        # # for layer_index, (layer_name, layer_data) in enumerate(zip(layers, all_data)):
        # #     plt.hist(layer_data, bins=num_bins, alpha=0.7, 
        # #              color=palette[layer_index], label=layer_name, density=density)

        # plt.yscale('log')

        # plt.title("Histogram of Neuron Activations Across All Layers")
        # plt.xlabel("Activation Values")
        # plt.ylabel("Log(Density)" if density else "Log(Count)")
        # plt.legend(title="Layers", loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.tight_layout()
        
        # combined_output_file = os.path.join(output_dir, "all_layers_histogram_log.png")
        # plt.savefig(combined_output_file, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"Combined histogram saved as {combined_output_file}")



import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os


def plot_histograms_std(h5_file_path, output_dir, num_bins=100):
    with h5py.File(h5_file_path, 'r') as f:
        layers = [key for key in f.keys() if key != 'image_indices']
        
        for layer_name in tqdm(layers, desc="Processing layers"):
            output_file = os.path.join(output_dir, f"{layer_name}_percentiles.txt")

            dataset = f[layer_name]
            layer_data = dataset[:].ravel()

            # Take a random subset of the data using linspace
            indices = np.linspace(0, len(layer_data) - 1, 100000000, dtype=int)
            layer_data = layer_data[indices]

            print("Calculating statistics")

            # Calculate mean and standard deviation
            mean = np.mean(layer_data)
            std_dev = np.std(layer_data)

            # Calculate stds

            # Create bins in terms of standard deviations
            sd_range = np.arange(-20, 20)  # From -2 to 40 SD, inclusive
            bins = mean + sd_range * std_dev

            # Calculate histogram
            print("Calculating histogram...")
            counts, bin_edges = np.histogram(layer_data, bins=bins)
            # bin_indices = np.digitize(layer_data, bins)
            # counts = np.bincount(bin_indices, minlength=len(bins))
            # bin_edges = bin_indices



            print("Calculating Cumulative Sum...")

            # # Calculate percentiles
            # total_count = len(layer_data)
            # cumulative_counts = np.cumsum(counts)
            # percentiles = cumulative_counts / total_count * 100

            # Prepare data for saving
            data = pd.DataFrame({
                'SD_start': sd_range[:-1],
                'SD_end': sd_range[1:],
                'Activation_start': bin_edges[:-1],
                'Activation_end': bin_edges[1:],
                'Count': counts,
                # 'Start_Percentile': np.concatenate(([0], percentiles[:-1])),
                # 'End_Percentile': percentiles
            })

            # Save data
            data.to_csv(output_file, index=False, float_format='%.4f')
            print(f"Percentile data for {layer_name} saved as {output_file}")

            # delete everything
            del layer_data, dataset, counts, bin_edges, data


# import numpy as np
# import h5py
# import os
# from tqdm import tqdm

# def calculate_stats(dataset, sample_size=100000):
#     total_size = dataset.shape[0]
#     if total_size <= sample_size:
#         indices = np.arange(total_size)
#     else:
#         indices = np.linspace(0, total_size - 1, sample_size, dtype=int)
    
#     samples = dataset[indices]
#     if len(samples.shape) > 1:
#         samples = samples.reshape(-1)
    
#     overall_mean = np.mean(samples)
#     overall_std = np.std(samples)
#     overall_min = np.min(samples)
#     overall_max = np.max(samples)
#     quartiles = np.percentile(samples, [25, 50, 75])
    
#     return overall_mean, overall_std, overall_min, overall_max, quartiles, len(samples)

# def process_layer(h5_file_path, layer_name, output_dir, sample_size=100000):
#     with h5py.File(h5_file_path, 'r') as f:
#         dataset = f[layer_name]
#         overall_mean, overall_std, overall_min, overall_max, quartiles, actual_sample_size = calculate_stats(dataset, sample_size)
    
#     output_file = os.path.join(output_dir, f"{layer_name}_analysis.txt")
    
#     with open(output_file, 'w') as txt_file:
#         txt_file.write(f"Analysis for {layer_name}\n\n")
#         txt_file.write(f"Sample Size: {actual_sample_size}\n")
#         txt_file.write(f"Overall Mean: {overall_mean:.4e}\n")
#         txt_file.write(f"Overall Standard Deviation: {overall_std:.4e}\n")
#         txt_file.write(f"Overall Activation Value Interval: [{overall_min:.4e}, {overall_max:.4e}]\n\n")
        
#         quartile_names = ["0-25%", "25-50%", "50-75%", "75-100%"]
#         quartile_bounds = [overall_min] + list(quartiles) + [overall_max]
        
#         for i, name in enumerate(quartile_names):
#             lower_bound = quartile_bounds[i]
#             upper_bound = quartile_bounds[i+1]
            
#             txt_file.write(f"Quartile: {name}\n")
#             txt_file.write(f"  Standard Deviation Interval: [{overall_mean - overall_std:.4e}, {overall_mean + overall_std:.4e}]\n")
#             txt_file.write(f"  Activation Value Interval: [{lower_bound:.4e}, {upper_bound:.4e}]\n\n")

#         txt_file.write("Quartile Boundaries:\n")
#         for i, q in enumerate([25, 50, 75]):
#             txt_file.write(f"  {q}th Percentile: {quartiles[i]:.4e}\n")

#     print(f"Analysis for {layer_name} written to {output_file}")

# def process_all_layers(h5_file_path, output_dir, sample_size=50000):
#     os.makedirs(output_dir, exist_ok=True)
    
#     with h5py.File(h5_file_path, 'r') as f:
#         layers = [key for key in f.keys() if key != 'image_indices']
    
#     for layer_name in tqdm(layers, desc="Processing layers"):
#         try:
#             process_layer(h5_file_path, layer_name, output_dir, sample_size)
#         except Exception as e:
#             print(f"An error occurred while processing {layer_name}: {e}")



def main():
    parser = argparse.ArgumentParser(description="Plot histogram of all neuron activations across all layers")
    parser.add_argument("--file_path", type=str, default= '/network/scratch/s/sonia.joseph/CLIP_AUDIT/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/val/blocks.{layer}.hook_mlp_out.h5', help="Path to the H5 file")
    parser.add_argument("--output", type=str, default="histograms", help="Output file name")
    parser.add_argument("--num_bins", type=int, default=400, help="Number of bins for the histogram")
    parser.add_argument("--density", action="store_true", help="Plot density instead of count")
    # either normal histogram or std histogram
    parser.add_argument("--std", action="store_true", help="Plot histogram in terms of standard deviations")

    args = parser.parse_args()

    if args.std:
        plot_histograms_std(args.file_path, args.output)
    else:
        plot_histograms(args.file_path, args.output, args.num_bins, args.density)

if __name__ == "__main__":
    main()