import h5py
import numpy as np

from clip_audit.utils.load_imagenet import load_imagenet, get_imagenet_names

import torch

# import transforms
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt

from vit_prisma.models.base_vit import HookedViT

# import einops
import einops


def sample_image_indices(file_path, activation_range, layer, neuron, n_samples=20):
    """
    Sample image indices for a given activation range, layer, and neuron.

    Parameters:
    - file_path (str): Path to the H5 file containing the activations
    - activation_range (tuple): (min_activation, max_activation)
    - layer (int): The layer number
    - neuron (int): The neuron number
    - n_samples (int): Number of samples to return (default 20)

    Returns:
    - list: Randomly sampled image indices
    """
    with h5py.File(file_path, 'r') as f:
        # Get the activation data for the specified layer
        layer_key = f'blocks.{layer}.hook_mlp_out'
        activations = f[layer_key][:, :, neuron]  # Shape: (n_images, sequence_length)

        # # Get the maximum activation for each image
        max_activations = np.max(activations, axis=1)

        # Find indices where activations are within the specified range
        min_act, max_act = activation_range
        valid_indices = np.where((activations >= min_act) & (activations <= max_act))[0]

        # If we have fewer valid indices than requested samples, return all of them
        if len(valid_indices) <= n_samples:
            sampled_indices = valid_indices
        else:
            # Randomly sample from the valid indices
            sampled_indices = np.random.choice(valid_indices, size=n_samples, replace=False)

        # Get the corresponding image indices
        image_indices = f['image_indices'][:]
        sampled_image_indices = image_indices[sampled_indices]

    return sampled_image_indices.tolist()


# get heatmaps of the top 20 images

torch.no_grad()
device = 'cuda'

def get_all_activations(
          
          image,
          model,
          hook_point_name,
          neuron_idx
): 
    image = image.to(device)
    _, cache = model.run_with_cache(image.unsqueeze(0), names_filter=[hook_point_name])

    post_reshaped = einops.rearrange(cache[hook_point_name], "batch seq d_mlp -> (batch seq) d_mlp")
    post_reshaped = post_reshaped[:,neuron_idx]
    # Compute activations (not from a fwd pass, but explicitly, by taking only the feature we want)
    # This code is copied from the first part of the 'forward' method of the AutoEncoder class
    # # sae_in =  post_reshaped - sparse_autoencoder.b_dec # Remove decoder bias as per Anthropic
    # acts = einops.einsum(
    #         sae_in,
    #         sparse_autoencoder.W_enc[:, feature_id],
    #         "x d_in, d_in -> x",
    #     )
    return post_reshaped 
     
def image_patch_heatmap(activation_values,image_size=224, pixel_num=7):
    activation_values = activation_values.detach().cpu().numpy()
    activation_values = activation_values[1:]
    activation_values = activation_values.reshape(pixel_num, pixel_num)

    # Create a heatmap overlay
    heatmap = np.zeros((image_size, image_size))
    patch_size = image_size // pixel_num

    for i in range(pixel_num):
        for j in range(pixel_num):
            heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values[i, j]

    return heatmap

def load_dataset(imagenet_path):
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
    dataloader = load_imagenet(imagenet_path, 'val', shuffle=False, transform=transform)
    dataset = dataloader.dataset
    return dataset

hook_point_name = 'blocks.{layer}.hook_mlp_out'
file_path = f'/network/scratch/s/sonia.joseph/CLIP_AUDIT/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/val/{hook_point_name}.h5'
imagenet_path = '/network/scratch/s/sonia.joseph/datasets/kaggle_datasets'
dataset = load_dataset(imagenet_path)


# Example usage:
# activation_range = (0, 40)  # Example range
neuron = 1  # Example neuron
layer = 11

# Get neuron intervals based on standard deviatoin
intervals = get_neuron_intervals()





# SAMPLE IMAGES

sampled_indices = sample_image_indices(file_path, activation_range, layer, neuron)
print(f"Sampled image indices: {sampled_indices}")

images = []
for idx in sampled_indices:
    image, label, _ = dataset[idx]
    images.append(image)

    
grid_size = int(np.ceil(np.sqrt(len(images))))
fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(15, 15))

fig.suptitle(f"Neuron {neuron}, Layer {layer}, Activation range {activation_range}")
for ax in axs.flatten():
    ax.axis('off')


model_name = 'wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M'
model = HookedViT.from_pretrained(model_name, is_clip=True, is_timm=False)
model.to(device)


for idx, image in enumerate(images):
    # image, label, _ = dataset[i]
    # image = image.unsqueeze(0)
    # plt.imshow(image)
    # plt.show()
    # # activation_values = get_all_activations(image, model)
    # heatmap = image_patch_heatmap(activation_values)
    # plt.imshow(heatmap)
    # plt.show()

    row = idx // grid_size
    col = idx % grid_size

    all_activations = get_all_activations(image, model, hook_point_name=f'blocks.{layer}.hook_mlp_out', neuron_idx=neuron)
    heatmap = image_patch_heatmap(all_activations)

    axs[row, col].imshow(image.permute(1, 2, 0))
    axs[row, col].imshow(heatmap, cmap='viridis', alpha=0.3)  # Overlaying the heatmap
    # axs[row, col].set_title(f"{label} {val.item():0.03f} {'class token!' if has_zero else ''}")  
    axs[row, col].axis('off')  

    # Save figure
    # plt.savefig('heatmap.png')

plt.tight_layout()
plt.savefig('sample_images.png')
print("Done")
