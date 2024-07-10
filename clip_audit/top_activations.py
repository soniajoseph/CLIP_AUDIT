import h5py
import numpy as np

def sample_activation_interval(h5_file_path, layer_index, neuron_index, interval_min, interval_max, num_samples=20):
    with h5py.File(h5_file_path, 'r') as f:
        layers = list(f.keys())
        if layer_index >= len(layers):
            raise ValueError(f"Layer index {layer_index} is out of range. File contains {len(layers)} layers.")
        
        layer_name = layers[layer_index]
        dataset = f[layer_name]
        
        # Get all activation values for this neuron
        activation_values = dataset[:, :, neuron_index].flatten()
        
        # Find indices where activations are within the specified interval
        interval_indices = np.where((activation_values >= interval_min) & (activation_values <= interval_max))[0]
        
        if len(interval_indices) == 0:
            print(f"No activations found in the interval [{interval_min}, {interval_max}]")
            return []
        
        # Randomly sample from these indices
        num_samples = min(num_samples, len(interval_indices))  # Ensure we don't try to sample more than available
        sampled_indices = np.random.choice(interval_indices, size=num_samples, replace=False)
        
        # Calculate the original image indices and token indices
        num_images, tokens_per_image, _ = dataset.shape
        image_indices = sampled_indices // tokens_per_image
        token_indices = sampled_indices % tokens_per_image
        
        # Create a list of (image_index, token_index, activation_value) tuples
        samples = [(int(img_idx), int(token_idx), float(activation_values[idx])) 
                   for idx, img_idx, token_idx in zip(sampled_indices, image_indices, token_indices)]
        
    return samples

# Usage example
file_path = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/val/blocks.{layer}.mlp.hook_post.h5'
layer_index = 0
neuron_index = 2
interval_min = 0.5
interval_max = 1.0
samples = sample_activation_interval(file_path, layer_index, neuron_index, interval_min, interval_max)

print(f"20 random samples for neuron {neuron_index} in layer {layer_index} with activations in [{interval_min}, {interval_max}]:")
for i, (img_idx, token_idx, activation) in enumerate(samples, 1):
    print(f"{i}. Image {img_idx}, Token {token_idx}: Activation = {activation:.4f}")