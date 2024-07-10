import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os

def get_activations(h5_file_path, output_dir):
    intervals = [(-np.inf, -40)]
    intervals.extend([(i, i+1) for i in range(-40, 40)])
    intervals.append((40, np.inf))

    with h5py.File(h5_file_path, 'r') as f:
        layers = [key for key in f.keys() if key != 'image_indices']
        
        all_results = []
        
        for layer_name in tqdm(layers, desc="Processing layers"):
            dataset = f[layer_name]
            layer_data = dataset[:]
            
            num_neurons = layer_data.shape[-1]
            
            for neuron in tqdm(range(num_neurons), desc="Processing neurons"):
                neuron_data = layer_data[..., neuron].ravel()
                
                mean = np.mean(neuron_data)
                std = np.std(neuron_data)
                
                # Calculate SD values
                sd_values = (neuron_data - mean) / std
                
                neuron_results = {'layer': layer_name, 'neuron': neuron}
                
                for start, end in intervals:
                    interval_mask = (sd_values > start) & (sd_values <= end)
                    interval_activations = neuron_data[interval_mask]
                    
                    if len(interval_activations) > 0:
                        neuron_results.update({
                            f'sd_{start}_{end}_min': start,
                            f'sd_{start}_{end}_max': end,
                            f'activation_{start}_{end}_min': np.min(interval_activations),
                            f'activation_{start}_{end}_max': np.max(interval_activations),
                            f'count_{start}_{end}': len(interval_activations)
                        })
                    else:
                        neuron_results.update({
                            f'sd_{start}_{end}_min': start,
                            f'sd_{start}_{end}_max': end,
                            f'activation_{start}_{end}_min': np.nan,
                            f'activation_{start}_{end}_max': np.nan,
                            f'count_{start}_{end}': 0
                        })
                
                all_results.append(neuron_results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save DataFrame to CSV
    output_file = os.path.join(output_dir, "all_neuron_activations_SD_intervals.csv")
    df.to_csv(output_file, index=False)
    print(f"Neuron activations saved to {output_file}")
    
    return df

def main():

    hook_point_name = 'blocks.{layer}.hook_mlp_out'
    file_path = f'/network/scratch/s/sonia.joseph/CLIP_AUDIT/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/val/{hook_point_name}.h5'
    parser = argparse.ArgumentParser(description="Get neuron activations across SD intervals")
    parser.add_argument("--file_path", type=str, default=file_path, required=False, help="Path to the H5 file")
    parser.add_argument("--output_dir", type=str, default="mlp.hook_out", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True) 
    df = get_activations(args.file_path, args.output_dir)
    
    # Print summary
    print("\nDataFrame Summary:")
    print(df.describe())
    print("\nDataFrame Head:")
    print(df.head())

if __name__ == "__main__":
    main()