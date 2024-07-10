import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse

def collect_layer_activations(h5_file_path, output_dir, intervals):
    with h5py.File(h5_file_path, 'r') as f:
        layers = [key for key in f.keys() if key != 'image_indices']
        
        results = []
        
        for layer_name in tqdm(layers, desc="Processing layers"):
            output_file = os.path.join(output_dir, "layer_activations.csv")
            
            dataset = f[layer_name]
            layer_data = dataset[:].ravel()  # Flatten all dimensions
            
            layer_results = {'layer': layer_name}
            
            percentiles = np.percentile(layer_data, [interval[1] for interval in intervals])
            
            for i, (start, end) in enumerate(intervals):
                if i == 0:
                    interval_data = layer_data[layer_data <= percentiles[i]]
                elif i == len(intervals) - 1:
                    interval_data = layer_data[layer_data > percentiles[i-1]]
                else:
                    interval_data = layer_data[(layer_data > percentiles[i-1]) & (layer_data <= percentiles[i])]
                
                layer_results[f'{start}-{end}_min'] = np.min(interval_data) if len(interval_data) > 0 else np.nan
                layer_results[f'{start}-{end}_max'] = np.max(interval_data) if len(interval_data) > 0 else np.nan
                layer_results[f'{start}-{end}_mean'] = np.mean(interval_data) if len(interval_data) > 0 else np.nan
                layer_results[f'{start}-{end}_std'] = np.std(interval_data) if len(interval_data) > 0 else np.nan
            
            results.append(layer_results)
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Layer activations saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Collect layer activations at specific intervals")
    parser.add_argument("--file_path", type=str, default='/network/scratch/s/sonia.joseph/CLIP_AUDIT/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/val/blocks.{layer}.hook_mlp_out.h5', help="Path to the H5 file")
    parser.add_argument("--output_dir", type=str, default='histograms/mlp.hook_out', help="Output directory")
    args = parser.parse_args()

    intervals = [
        (0, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99),
        (99, 99.9), (99.9, 99.99), (99.99, 99.999), (99.999, 99.9999),
        (99.9999, 99.99999), (99.99999, 100)
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    collect_layer_activations(args.file_path, args.output_dir, intervals)

if __name__ == "__main__":
    main()